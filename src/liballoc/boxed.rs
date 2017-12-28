// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pointer type for heap allocation.
//!
//! `Box<T>`, casually referred to as a 'box', provides the simplest form of
//! heap allocation in Rust. Boxes provide ownership for this allocation, and
//! drop their contents when they go out of scope.
//!
//! # Examples
//!
//! Creating a box:
//!
//! ```
//! let x = Box::new(5);
//! ```
//!
//! Creating a recursive data structure:
//!
//! ```
//! #[derive(Debug)]
//! enum List<T> {
//!     Cons(T, Box<List<T>>),
//!     Nil,
//! }
//!
//! fn main() {
//!     let list: List<i32> = List::Cons(1, Box::new(List::Cons(2, Box::new(List::Nil))));
//!     println!("{:?}", list);
//! }
//! ```
//!
//! This will print `Cons(1, Cons(2, Nil))`.
//!
//! Recursive structures must be boxed, because if the definition of `Cons`
//! looked like this:
//!
//! ```compile_fail,E0072
//! # enum List<T> {
//! Cons(T, List<T>),
//! # }
//! ```
//!
//! It wouldn't work. This is because the size of a `List` depends on how many
//! elements are in the list, and so we don't know how much memory to allocate
//! for a `Cons`. By introducing a `Box`, which has a defined size, we know how
//! big `Cons` needs to be.

#![stable(feature = "rust1", since = "1.0.0")]

use abort_adapter::AbortAdapter;
use heap::{Heap, Layout, Alloc};
use raw_vec::RawVec;

use core::any::Any;
use core::borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{self, Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::{self, Unsize};
use core::mem;
use core::ops::{CoerceUnsized, Deref, DerefMut, Generator, GeneratorState};
use core::ops::{BoxPlace, Boxed, InPlace, Place, Placer};
use core::ptr::{self, Unique};
use core::convert::From;
use str::from_boxed_utf8_unchecked;

/// A value that represents the heap. This is the default place that the `box`
/// keyword allocates into when no place is supplied.
///
/// The following two examples are equivalent:
///
/// ```
/// #![feature(box_heap)]
///
/// #![feature(box_syntax, placement_in_syntax)]
/// use std::boxed::HEAP;
///
/// fn main() {
///     let foo: Box<i32> = in HEAP { 5 };
///     let foo = box 5;
/// }
/// ```
#[unstable(feature = "box_heap",
           reason = "may be renamed; uncertain about custom allocator design",
           issue = "27779")]
pub const HEAP: AbortAdapter<Heap> = AbortAdapter(Heap);

/// A pointer type for heap allocation.
///
/// See the [module-level documentation](../../std/boxed/index.html) for more.
#[lang = "owned_box"]
#[fundamental]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Box<T: ?Sized, A: Alloc = AbortAdapter<Heap>>(Unique<T>, A);

#[cfg(stage0)]
#[lang = "box_free"]
#[inline]
pub(crate) unsafe fn old_box_free<T: ?Sized>(ptr: *mut T) {
    box_free(Unique::new_unchecked(ptr), HEAP)
}

#[cfg_attr(not(any(test, stage0)), lang = "box_free")]
#[inline]
// Invoked by the compiler to deallocate the storage of `Box<T>`,
// after the owned `T` value on the heap has already been dropped.
// NB: the generics should be the same as for `Box`, and the
// argument types should match the fields in `struct Box`.
pub(crate) unsafe fn box_free<T: ?Sized, A: Alloc>(ptr: Unique<T>, mut a: A) {
    box_free_worker(ptr, &mut a)
}

#[inline]
pub(crate) unsafe fn box_free_worker<T: ?Sized, A: Alloc>(ptr: Unique<T>, a: &mut A) {
    let ptr = ptr.as_ptr();
    let size = mem::size_of_val(&*ptr);
    let align = mem::align_of_val(&*ptr);
    // We do not allocate for Box<T> when T is ZST, so deallocation is also not necessary.
    if size != 0 {
        let layout = Layout::from_size_align_unchecked(size, align);
        a.dealloc(ptr as *mut u8, layout);
    }
}

/// `IntermediateBox` represents uninitialized backing storage for `Box`.
///
/// FIXME (pnkfelix): Ideally we would just reuse `Box<T, A>` instead of
/// introducing a separate `IntermediateBox<T, A>`; but then you hit
/// issues when you e.g. attempt to destructure an instance of `Box`,
/// since it is a lang item and so it gets special handling by the
/// compiler.  Easier just to make this parallel type for now.
///
/// FIXME (pnkfelix): Currently the `box` protocol only supports
/// creating instances of sized types. This IntermediateBox is
/// designed to be forward-compatible with a future protocol that
/// supports creating instances of unsized types; that is why the type
/// parameter has the `?Sized` generalization marker, and is also why
/// this carries an explicit size. However, it probably does not need
/// to carry the explicit alignment; that is just a work-around for
/// the fact that the `align_of` intrinsic currently requires the
/// input type to be Sized (which I do not think is strictly
/// necessary).
#[unstable(feature = "placement_in",
           reason = "placement box design is still being worked out.",
           issue = "27779")]
#[allow(missing_debug_implementations)]
pub struct IntermediateBox<T: ?Sized, A: Alloc = AbortAdapter<Heap>> {
    ptr: *mut u8,
    layout: Layout,
    marker: marker::PhantomData<*mut T>,
    alloc: A,
}

#[unstable(feature = "placement_in",
           reason = "placement box design is still being worked out.",
           issue = "27779")]
impl<T, A: Alloc> Place<T> for IntermediateBox<T, A> {
    fn pointer(&mut self) -> *mut T {
        self.ptr as *mut T
    }
}

unsafe fn finalize<T, A: Alloc>(mut b: IntermediateBox<T, A>) -> Box<T, A> {
    let ptr = b.ptr as *mut T;
    let alloc = {
        let mut a = mem::uninitialized();
        mem::swap(&mut a, &mut b.alloc);
        a
    };
    mem::forget(b);
    Box::<T, A>::from_raw_in(ptr, alloc)
}

fn make_place<T, A: Alloc<Err=!> + Default>() -> IntermediateBox<T, A> {
    let Ok(b) = make_place_alloc(Default::default());
    b
}

#[unstable(feature = "allocator_api", issue = "32838")]
fn make_place_alloc<T, A: Alloc>(mut alloc: A) -> Result<IntermediateBox<T, A>, A::Err> {
    let layout = Layout::new::<T>();

    let p = if layout.size() == 0 {
        mem::align_of::<T>() as *mut u8
    } else {
        unsafe {
            alloc.alloc(layout.clone())?
        }
    };

    Ok(IntermediateBox {
        ptr: p,
        layout,
        marker: marker::PhantomData,
        alloc: alloc
    })
}

#[unstable(feature = "placement_in",
           reason = "placement box design is still being worked out.",
           issue = "27779")]
impl<T, A: Default + Alloc<Err=!>> BoxPlace<T> for IntermediateBox<T, A> {
    fn make_place() -> IntermediateBox<T, A> {
        make_place()
    }
}

#[unstable(feature = "placement_in",
           reason = "placement box design is still being worked out.",
           issue = "27779")]
impl<T, A: Alloc> InPlace<T> for IntermediateBox<T, A> {
    type Owner = Box<T, A>;
    unsafe fn finalize(self) -> Box<T, A> {
        finalize(self)
    }
}

#[unstable(feature = "placement_new_protocol", issue = "27779")]
impl<T, A: Default + Alloc<Err=!>> Boxed for Box<T, A> {
    type Data = T;
    type Place = IntermediateBox<T, A>;
    unsafe fn finalize(b: IntermediateBox<T, A>) -> Box<T, A> {
        finalize(b)
    }
}

#[unstable(feature = "placement_in",
           reason = "placement box design is still being worked out.",
           issue = "27779")]
impl<T> Placer<T> for AbortAdapter<Heap> {
    type Place = IntermediateBox<T, AbortAdapter<Heap>>;

    fn make_place(self) -> Self::Place {
        make_place()
    }
}

// WTF coherence
//
//#[unstable(feature = "placement_in",
//           reason = "placement box design is still being worked out.",
//           issue = "27779")]
//impl<T, A: Alloc<Err=!> + Default> Placer<T> for A {
//    type Place = IntermediateBox<T, A>;
//
//    fn make_place(self) -> Self::Place {
//        make_place()
//    }
//}

#[unstable(feature = "placement_in",
           reason = "placement box design is still being worked out.",
           issue = "27779")]
impl<T: ?Sized, A: Alloc> Drop for IntermediateBox<T, A> {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe {
                self.alloc.dealloc(self.ptr, self.layout.clone())
            }
        }
    }
}

impl<T> Box<T> {
    /// Allocates memory on the heap and then places `x` into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = Box::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline(always)]
    pub fn new(x: T) -> Box<T> {
        box x
    }
}

impl<T, A: Alloc> Box<T, A> {
    /// Allocates memory on the heap and then places `x` into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = Box::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline(always)]
    pub fn new_in(x: T, a: A) -> Result<Box<T, A>, A::Err> {
        let mut place = make_place_alloc(a)?;
        let raw_place = Place::pointer(&mut place);
        // TODO: Make placement API support result.
        Ok(unsafe {
            ptr::write(raw_place, x);
            InPlace::finalize(place)
        })
    }
}


impl<T: ?Sized> Box<T> {
    /// Constructs a box from a raw pointer.
    ///
    /// After calling this function, the raw pointer is owned by the
    /// resulting `Box`. Specifically, the `Box` destructor will call
    /// the destructor of `T` and free the allocated memory. Since the
    /// way `Box` allocates and releases memory is unspecified, the
    /// only valid pointer to pass to this function is the one taken
    /// from another `Box` via the [`Box::into_raw`] function.
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    ///
    /// [`Box::into_raw`]: struct.Box.html#method.into_raw
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let ptr = Box::into_raw(x);
    /// let x = unsafe { Box::from_raw(ptr) };
    /// ```
    #[stable(feature = "box_raw", since = "1.4.0")]
    #[inline]
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        Box::from_unique_alloc(Unique::new_unchecked(raw), Default::default())
    }

    /// Constructs a `Box` from a `Unique<T>` pointer.
    ///
    /// After calling this function, the memory is owned by a `Box` and `T` can
    /// then be destroyed and released upon drop.
    ///
    /// # Safety
    ///
    /// A `Unique<T>` can be safely created via [`Unique::new`] and thus doesn't
    /// necessarily own the data pointed to nor is the data guaranteed to live
    /// as long as the pointer.
    ///
    /// [`Unique::new`]: ../../core/ptr/struct.Unique.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique)]
    ///
    /// fn main() {
    ///     let x = Box::new(5);
    ///     let ptr = Box::into_unique(x);
    ///     let x = unsafe { Box::from_unique(ptr) };
    /// }
    /// ```
    #[unstable(feature = "unique", reason = "needs an RFC to flesh out design",
               issue = "27730")]
    #[inline]
    pub unsafe fn from_unique(u: Unique<T>) -> Self {
        Box::from_unique_alloc(u, Default::default())
    }
}

#[unstable(feature = "allocator_api", issue = "32838")]
impl<T: ?Sized, A: Alloc> Box<T, A> {
    #[inline]
    pub unsafe fn from_raw_in(raw: *mut T, a: A) -> Self {
        Box::from_unique_alloc(Unique::new_unchecked(raw), a)
    }

    #[unstable(feature = "unique", reason = "needs an RFC to flesh out design",
               issue = "27730")]
    #[inline]
    pub unsafe fn from_unique_alloc(u: Unique<T>, a: A) -> Self {
        Box(u, a)
    }

    /// Consumes the `Box`, returning the wrapped raw pointer.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `Box`. In particular, the
    /// caller should properly destroy `T` and release the memory. The
    /// proper way to do so is to convert the raw pointer back into a
    /// `Box` with the [`Box::from_raw`] function.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::into_raw(b)` instead of `b.into_raw()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`Box::from_raw`]: struct.Box.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let ptr = Box::into_raw(x);
    /// ```
    #[stable(feature = "box_raw", since = "1.4.0")]
    #[inline]
    pub fn into_raw(b: Self) -> *mut T {
        Box::into_unique(b).as_ptr()
    }

    /// Consumes the `Box`, returning the wrapped pointer as `Unique<T>`.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `Box`. In particular, the
    /// caller should properly destroy `T` and release the memory. The
    /// proper way to do so is to either convert the `Unique<T>` pointer:
    ///
    /// - Into a `Box` with the [`Box::from_unique`] function.
    ///
    /// - Into a raw pointer and back into a `Box` with the [`Box::from_raw`]
    ///   function.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::into_unique(b)` instead of `b.into_unique()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`Box::from_unique`]: struct.Box.html#method.from_unique
    /// [`Box::from_raw`]: struct.Box.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique)]
    ///
    /// fn main() {
    ///     let x = Box::new(5);
    ///     let ptr = Box::into_unique(x);
    /// }
    /// ```
    #[unstable(feature = "unique", reason = "needs an RFC to flesh out design",
               issue = "27730")]
    #[inline]
    pub fn into_unique(b: Self) -> Unique<T> {
        let unique = b.0;
        mem::forget(b);
        unique
    }


    #[unstable(feature = "unique", reason = "needs an RFC to flesh out design",
               issue = "27730")]
    #[inline]
    pub fn into_both(mut b: Self) -> (Unique<T>, A) {
        let unique = b.0;
        let alloc = unsafe {
            let mut a = mem::uninitialized();
            mem::swap(&mut a, &mut b.1);
            a
        };
        mem::forget(b);
        (unique, alloc)
    }

    /// Consumes and leaks the `Box`, returning a mutable reference,
    /// `&'a mut T`. Here, the lifetime `'a` may be chosen to be `'static`.
    ///
    /// This function is mainly useful for data that lives for the remainder of
    /// the program's life. Dropping the returned reference will cause a memory
    /// leak. If this is not acceptable, the reference should first be wrapped
    /// with the [`Box::from_raw`] function producing a `Box`. This `Box` can
    /// then be dropped which will properly destroy `T` and release the
    /// allocated memory.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::leak(b)` instead of `b.leak()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`Box::from_raw`]: struct.Box.html#method.from_raw
    ///
    /// # Examples
    ///
    /// Simple usage:
    ///
    /// ```
    /// #![feature(box_leak)]
    ///
    /// fn main() {
    ///     let x = Box::new(41);
    ///     let static_ref: &'static mut usize = Box::leak(x);
    ///     *static_ref += 1;
    ///     assert_eq!(*static_ref, 42);
    /// }
    /// ```
    ///
    /// Unsized data:
    ///
    /// ```
    /// #![feature(box_leak)]
    ///
    /// fn main() {
    ///     let x = vec![1, 2, 3].into_boxed_slice();
    ///     let static_ref = Box::leak(x);
    ///     static_ref[0] = 4;
    ///     assert_eq!(*static_ref, [4, 2, 3]);
    /// }
    /// ```
    #[unstable(feature = "box_leak", reason = "needs an FCP to stabilize",
               issue = "46179")]
    #[inline]
    pub fn leak<'a>(b: Self) -> &'a mut T
    where
        T: 'a // Technically not needed, but kept to be explicit.
    {
        unsafe { &mut *Box::into_raw(b) }
    }
}


#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T: ?Sized, A: Alloc> Drop for Box<T, A> {
    fn drop(&mut self) {
        // FIXME: Do nothing, drop is currently performed by compiler.
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default, A: Alloc<Err=!> + Default> Default for Box<T, A> {
    /// Creates a `Box<T, A>`, with the `Default` value for T.
    fn default() -> Box<T, A> {
        let Ok(b) = Box::new_in(Default::default(), Default::default());
        b
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Alloc<Err=!> + Default> Default for Box<[T], A> {
    fn default() -> Box<[T], A> {
        let Ok(b) = Box::<[T; 0], A>::new_in([], Default::default());
        b
    }
}

#[stable(feature = "default_box_extra", since = "1.17.0")]
impl Default for Box<str> {
    fn default() -> Box<str> {
        unsafe { from_boxed_utf8_unchecked(Default::default()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone, A: Alloc<Err=!> + Clone> Clone for Box<T, A> {
    /// Returns a new box with a `clone()` of this box's contents.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let y = x.clone();
    /// ```
    #[rustfmt_skip]
    #[inline]
    fn clone(&self) -> Self {
        let Ok(b) = Box::new_in((**self).clone(), self.1.clone());
        b
    }
    /// Copies `source`'s contents into `self` without creating a new allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let mut y = Box::new(10);
    ///
    /// y.clone_from(&x);
    ///
    /// assert_eq!(*y, 5);
    /// ```
    #[inline]
    fn clone_from(&mut self, source: &Self) {
        (**self).clone_from(&(**source));
    }
}


#[stable(feature = "box_slice_clone", since = "1.3.0")]
impl Clone for Box<str> {
    fn clone(&self) -> Self {
        let len = self.len();
        let buf = RawVec::with_capacity(len);
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), buf.ptr(), len);
            from_boxed_utf8_unchecked(buf.into_box())
        }
    }
}

/// Just the contents are compared, the allocator is ignored
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq, A: Alloc> PartialEq for Box<T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        PartialEq::ne(&**self, &**other)
    }
}
/// Just the contents are compared, the allocator is ignored
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd, A: Alloc> PartialOrd for Box<T, A> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    #[inline]
    fn lt(&self, other: &Self) -> bool {
        PartialOrd::lt(&**self, &**other)
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        PartialOrd::le(&**self, &**other)
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        PartialOrd::ge(&**self, &**other)
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}
/// Just the contents are compared, the allocator is ignored
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord, A: Alloc> Ord for Box<T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
/// Just the contents are compared, the allocator is ignored
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq, A: Alloc> Eq for Box<T, A> {}

/// Just the contents are compared, the allocator is ignored
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Hash, A: Alloc> Hash for Box<T, A> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

/// Just the contents are compared, the allocator is ignored
#[stable(feature = "indirect_hasher_impl", since = "1.22.0")]
impl<T: ?Sized + Hasher, A: Alloc> Hasher for Box<T, A> {
    fn finish(&self) -> u64 {
        (**self).finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        (**self).write(bytes)
    }
    fn write_u8(&mut self, i: u8) {
        (**self).write_u8(i)
    }
    fn write_u16(&mut self, i: u16) {
        (**self).write_u16(i)
    }
    fn write_u32(&mut self, i: u32) {
        (**self).write_u32(i)
    }
    fn write_u64(&mut self, i: u64) {
        (**self).write_u64(i)
    }
    fn write_u128(&mut self, i: u128) {
        (**self).write_u128(i)
    }
    fn write_usize(&mut self, i: usize) {
        (**self).write_usize(i)
    }
    fn write_i8(&mut self, i: i8) {
        (**self).write_i8(i)
    }
    fn write_i16(&mut self, i: i16) {
        (**self).write_i16(i)
    }
    fn write_i32(&mut self, i: i32) {
        (**self).write_i32(i)
    }
    fn write_i64(&mut self, i: i64) {
        (**self).write_i64(i)
    }
    fn write_i128(&mut self, i: i128) {
        (**self).write_i128(i)
    }
    fn write_isize(&mut self, i: isize) {
        (**self).write_isize(i)
    }
}

#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T, A: Alloc<Err=!> + Default> From<T> for Box<T, A> {
    fn from(t: T) -> Self {
        let Ok(b) = Box::new_in(t, Default::default());
        b
    }
}

#[stable(feature = "box_from_slice", since = "1.17.0")]
impl<'a, T: Copy> From<&'a [T]> for Box<[T]> {
    fn from(slice: &'a [T]) -> Box<[T]> {
        let mut boxed = unsafe { RawVec::with_capacity(slice.len()).into_box() };
        boxed.copy_from_slice(slice);
        boxed
    }
}

#[stable(feature = "box_from_slice", since = "1.17.0")]
impl<'a> From<&'a str> for Box<str> {
    fn from(s: &'a str) -> Box<str> {
        unsafe { from_boxed_utf8_unchecked(Box::from(s.as_bytes())) }
    }
}

#[stable(feature = "boxed_str_conv", since = "1.19.0")]
impl From<Box<str>> for Box<[u8]> {
    fn from(s: Box<str>) -> Self {
        unsafe { Box::from_raw(Box::into_raw(s) as *mut [u8]) }
    }
}

impl<A: Alloc> Box<Any, A> {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Attempt to downcast the box to a concrete type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn print_if_string(value: Box<Any>) {
    ///     if let Ok(string) = value.downcast::<String>() {
    ///         println!("String ({}): {}", string.len(), string);
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let my_string = "Hello World".to_string();
    ///     print_if_string(Box::new(my_string));
    ///     print_if_string(Box::new(0i8));
    /// }
    /// ```
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Box<Any, A>> {
        if self.is::<T>() {
            unsafe {
                let (u, a) = Box::into_both(self);
                Ok(Box::from_raw_in(u.as_ptr() as *mut T, a))
            }
        } else {
            Err(self)
        }
    }
}

impl<A: Alloc> Box<Any + Send, A> {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Attempt to downcast the box to a concrete type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn print_if_string(value: Box<Any + Send>) {
    ///     if let Ok(string) = value.downcast::<String>() {
    ///         println!("String ({}): {}", string.len(), string);
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let my_string = "Hello World".to_string();
    ///     print_if_string(Box::new(my_string));
    ///     print_if_string(Box::new(0i8));
    /// }
    /// ```
    pub fn downcast<T: Any>(self) -> Result<Box<T, A>, Box<Any + Send, A>> {
        <Box<Any, A>>::downcast(self).map_err(|s| unsafe {
            let (u, a) = Box::into_both(s);
            // reapply the Send marker
            Box::from_raw_in(u.as_ptr() as *mut (Any + Send), a)
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Display + ?Sized, A: Alloc> fmt::Display for Box<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug + ?Sized, A: Alloc> fmt::Debug for Box<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Alloc> fmt::Pointer for Box<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // It's not possible to extract the inner Uniq directly from the Box,
        // instead we cast it to a *const which aliases the Unique
        let ptr: *const T = &**self;
        fmt::Pointer::fmt(&ptr, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Alloc> Deref for Box<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        &**self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Alloc> DerefMut for Box<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized, A: Alloc> Iterator for Box<I, A> {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> {
        (**self).next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        (**self).nth(n)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator + ?Sized, A: Alloc> DoubleEndedIterator for Box<I, A> {
    fn next_back(&mut self) -> Option<I::Item> {
        (**self).next_back()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator + ?Sized, A: Alloc> ExactSizeIterator for Box<I, A> {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<I: FusedIterator + ?Sized, A: Alloc> FusedIterator for Box<I, A> {}


/// `FnBox` is a version of the `FnOnce` intended for use with boxed
/// closure objects. The idea is that where one would normally store a
/// `Box<FnOnce()>` in a data structure, you should use
/// `Box<FnBox()>`. The two traits behave essentially the same, except
/// that a `FnBox` closure can only be called if it is boxed. (Note
/// that `FnBox` may be deprecated in the future if `Box<FnOnce()>`
/// closures become directly usable.)
///
/// # Examples
///
/// Here is a snippet of code which creates a hashmap full of boxed
/// once closures and then removes them one by one, calling each
/// closure as it is removed. Note that the type of the closures
/// stored in the map is `Box<FnBox() -> i32>` and not `Box<FnOnce()
/// -> i32>`.
///
/// ```
/// #![feature(fnbox)]
///
/// use std::boxed::FnBox;
/// use std::collections::HashMap;
///
/// fn make_map() -> HashMap<i32, Box<FnBox() -> i32>> {
///     let mut map: HashMap<i32, Box<FnBox() -> i32>> = HashMap::new();
///     map.insert(1, Box::new(|| 22));
///     map.insert(2, Box::new(|| 44));
///     map
/// }
///
/// fn main() {
///     let mut map = make_map();
///     for i in &[1, 2] {
///         let f = map.remove(&i).unwrap();
///         assert_eq!(f(), i * 22);
///     }
/// }
/// ```
#[rustc_paren_sugar]
#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
pub trait FnBox<As, A: Alloc = AbortAdapter<Heap>> {
    type Output;

    fn call_box(self: Box<Self, A>, args: As) -> Self::Output;
}

#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
impl<A, F> FnBox<A> for F
    where F: FnOnce<A>
{
    type Output = F::Output;

    fn call_box(self: Box<F>, args: A) -> F::Output {
        self.call_once(args)
    }
}

#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
impl<'a, A, R> FnOnce<A> for Box<FnBox<A, Output = R> + 'a> {
    type Output = R;

    extern "rust-call" fn call_once(self, args: A) -> R {
        self.call_box(args)
    }
}

#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
impl<'a, A, R> FnOnce<A> for Box<FnBox<A, Output = R> + Send + 'a> {
    type Output = R;

    extern "rust-call" fn call_once(self, args: A) -> R {
        self.call_box(args)
    }
}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Alloc>
    CoerceUnsized<Box<U, A>> for Box<T, A> {}

#[stable(feature = "box_slice_clone", since = "1.3.0")]
impl<T: Clone> Clone for Box<[T]> {
    fn clone(&self) -> Self {
        let mut new = BoxBuilder {
            data: RawVec::with_capacity(self.len()),
            len: 0,
        };

        let mut target = new.data.ptr();

        for item in self.iter() {
            unsafe {
                ptr::write(target, item.clone());
                target = target.offset(1);
            };

            new.len += 1;
        }

        return unsafe { new.into_box() };

        // Helper type for responding to panics correctly.
        struct BoxBuilder<T> {
            data: RawVec<T>,
            len: usize,
        }

        impl<T> BoxBuilder<T> {
            unsafe fn into_box(self) -> Box<[T]> {
                let raw = ptr::read(&self.data);
                mem::forget(self);
                raw.into_box()
            }
        }

        impl<T> Drop for BoxBuilder<T> {
            fn drop(&mut self) {
                let mut data = self.data.ptr();
                let max = unsafe { data.offset(self.len as isize) };

                while data != max {
                    unsafe {
                        ptr::read(data);
                        data = data.offset(1);
                    }
                }
            }
        }
    }
}

#[stable(feature = "box_borrow", since = "1.1.0")]
impl<T: ?Sized, A: Alloc> borrow::Borrow<T> for Box<T, A> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[stable(feature = "box_borrow", since = "1.1.0")]
impl<T: ?Sized, A: Alloc> borrow::BorrowMut<T> for Box<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized, A: Alloc> AsRef<T> for Box<T, A> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized, A: Alloc> AsMut<T> for Box<T, A> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[unstable(feature = "generator_trait", issue = "43122")]
impl<T, A: Alloc> Generator for Box<T, A>
    where T: Generator + ?Sized
{
    type Yield = T::Yield;
    type Return = T::Return;
    fn resume(&mut self) -> GeneratorState<Self::Yield, Self::Return> {
        (**self).resume()
    }
}
