---
layout: post
author: Andrew Abok
title: Immutability Doesn't Mean Copying Everything
date: 2026-07-13 00:00:00 +0300
description: Persistent data structures aren't efficient because they copy less data. They're efficient because recursion lets them avoid copying almost all of it.
categories: [Software development]
tags: [F#, .NET]
math: true
---

## Introduction

I havce been reading Okasaki's book on purely functional data structures,Having worked with diffretn proragmming paradigms i thought  i already understood the trade offs properly and how they actually work.I had simplified immutable data structures to one mental model:

- Mutable data structures modify memeory in place.
- Immutable data structures copy everything.

Well it turns out the first statement is correct and the second isn't.

## The Obvious Implementaion

Let's do this ilustration in F#,suppose we have a simple list

```fsharp
let xs = [1;2;3;4]
```

Just pause for a second and actually answer this: what exactly gets copied?

Imaguine that we modify the third element,picturing this you'd image or lets say i imagined:

$$
[1;2;3;4] \rightarrow copy \rightarrow [1;2;7;4]
$$

From the very defition of immutability every node should be copied, that would ceartainly preserve immutability,but programmatically it would also be terribly inefficient and i bet if immutable collections really behaved like this, they would probably never be practical.

Fortunately, if you read a bit on persistence in purely functional data structures, you'll find out that they don't. In fact, the F# source code for `Map.add` shows exactly how this works in practice.

## The surprising implementaion

I'll start by wrting a simple update function.

```fsharp
let rec update index value list = 
    match index,list with
    |_,[] ->
       invalidArg "index" "Out of range"
    |0,_::xs -> 
          value::xs
    |i,x::xs ->
        x::update(i-1) value xs
```

Now if we evaluate

```fsharp
let xs = [1;2;3;4;5]
let ys = update 2 7 xs
```

At first glance it looks recursive but something subtle happens, The recursion only rebuilds the nodes that lie on the path to the modification.
Conceptually the memory now looks like this.

$$
\begin{aligned}
\text{xs } 1 \rightarrow 2 \rightarrow 3 \rightarrow 4 \rightarrow 5 \\
\text{ys } 1' \rightarrow 2' \rightarrow 7 \rightarrow 4 \rightarrow 5
\end{aligned}
$$

Only the first two nodes and the modifed node are new while the last two nodes are literally the same objects, nothing copied.Both lists are simply pointing to them and both the new and the orginall lists exists and the most important thing here is that they are sharing memeory safely.

This mkaes the first indispensible point here, that the key idea behid persistent data structures is that copy only what changed and share everything else.
A better implemetnaion is teh `Add` function in Map in Fsharp here is a ref to the [source code](https://github.com/dotnet/fsharp/blob/527a6c3c19827f0801197b76e9e5480e9e7562ab/src/FSharp.Core/map.fs#L141) shows it beautifully.

```FSharp
let rec add (comparer: IComparer<'Key>) k (v: 'Value) (m: MapTree<'Key, 'Value>) =
    
    if c < 0 then
        rebalance (add comparer k v mn.Left) mn.Key mn.Value mn.Right
        //         ^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^
        //         Only this path changes           This is SHARED
```

From the F# source, notice how the add function recurses only on the affected side:

- If inserting on the left: `rebalance (add ... mn.Left) ... mn.Right`
- The right subtree `(mn.Right)` is passed through unchanged  this is teh Shared bit.

## Sharing is the optimization

When you first encounter immutable data structures you might focus on the copying (and that is rightfully so).But after reading purely functional data structure  i understood that the focu is on the sharing, and sharing here is not an implementation detail but rather it is the optimisation.Without sharing, persistence would be prohibitively expensive given we'd have to do a whole copy of the list for example in F# but with sharing, most updates allocate surprisingly little memory and this leaves the majority of the structure remains untouched.

## Recurive trees

The list example convinced me that immutable data structures don't copy everything, but it still left me with one question.

After all, the update function was recursive. Surely every recursive call was creating another copy of the list. If recursion keeps rebuilding the structure, why doesn't the whole thing get copied?

Then in the persistent data structure section he used trees, then for my own understanding i decided to make the switch from lists to trees.

Lists are linear every node only points to the next node. Binary search trees are more interesting because every node has two children. They make it much easier to see what is being rebuilt and, perhaps more importantly, what isn't.

Let's look at how F# implements this in the actual `map.fs` [source code](https://github.com/dotnet/fsharp/blob/5928e91b5f701586690562ce10bd639357fff50b/src/FSharp.Core/map.fs#L25):

```Fsharp
// From map.fs - the internal tree type
type internal MapTreeNode<'Key, 'Value>
    (k: 'Key, v: 'Value, left: MapTree<'Key, 'Value>, right: MapTree<'Key, 'Value>, h: int) =
    inherit MapTree<'Key, 'Value>(k, v, h)
    member _.Left = left
    member _.Right = right
```

The add function uses this structure to rebuild only the path (Again this is borrowed from the F# [source code](https://github.com/dotnet/fsharp/blob/5928e91b5f701586690562ce10bd639357fff50b/src/FSharp.Core/map.fs#L141)):

```Fsharp
//the recursive add function as implemented
let rec add (comparer: IComparer<'Key>) k (v: 'Value) (m: MapTree<'Key, 'Value>) =
    if isEmpty m then
        MapTree(k, v)  //This is the base case where  - create ONE new leaf
    else
        let c = comparer.Compare(k, m.Key)
        if m.Height = 1 then
            // Leaf node handling
            if c < 0 then MapTreeNode(k, v, empty, m, 2)
            // ...
        else
            let mn = asNode m
            if c < 0 then
                rebalance (add comparer k v mn.Left) mn.Key mn.Value mn.Right
                // Only the left path is rebuilt
            else
                rebalance mn.Left mn.Key mn.Value (add comparer k v mn.Right)
                // Only the right path is rebuilt
```

The insertion algorithm looks almost identical to the one you'll find in any algorithms textbook.

The first time I looked at this code, I completely missed what it was doing.There isn't a single line that mentions persistence.Nothing talks about structural sharing.There isn't even a comment saying "reuse the existing nodes." Yet that's exactly what the function does.

It took me a while to realise that the answer had very little to do with trees, and everything to do with recursion.Suppose we're inserting a value that belongs somewhere in the left subtree. Eventually the function reaches this line.

```fsharp
rebalance (add comparer k v mn.Left) mn.Key mn.Value mn.Right
```

At first glance it looks like we're rebuilding the entire tree but we're not, whats fundamentally happening is that we're quietly making a very bold assumption.We assume that this expression `add comparer k v mn.Left` has already produced a perfectly valid subtree.We don't care how many recursive calls it made nor how many nodes it created furthermore even how deep the tree was doesn't actually bother us,We simply trust that the subtree it returned is correct.Once we make that assumption, our job becomes surprisingly small all we need to do is to construct one new node and that's it.

The new node points to the updated left subtree, reuses the existing right subtree, and returns.The recursive call solved the difficult part and the current stack frame performs only the final step.

Reading Dao's functional data structure book, I came across a sentence that completely changed how I think about recursion:

> Recursion is based on suspension of disbelief. You tentatively assume that you already know how to solve the problem. Then you ask yourself: how would I perform > the last step if everything else had already been solved?

That is exactly what is happening here.Every recursive call returns an updated subtree.Every stack frame reconnects one node and nothing more.

Now imagine inserting 5 into this tree.

        8
      /   \
     4     12
    / \    / \
   2   6 10 14

The search path is simple `8 → 4 → 6`

Those are the only nodes that are rebuilt and everything else is reused.The resulting picture looks more like this.

Original tree

        8
      /   \
     4     12
    / \    / \
   2   6 10 14

Updated tree

        8'
      /    \
     4'     12
    / \     / \
   2   6'  10 14
      /
     5

Notice something subtle here,The node containing 12 wasn't copied and neither was the subtree rooted at 10 nor the one rooted at 14.They exist only once in memory and both trees simply point to them.At that moment the earlier list example suddenly clicked for me.The algorithm was never copying a tree rather it was merely copying the path that led to the modification while everything else was simply shared.That's the real optimisation behind persistent data structures its not faster copying.

Avoiding almost all copying in the first place.

## Copying paths

After looking at the recurrion implemntaion and the persistence data structures this completely changed how I think about immutable data structures.
We're not copying objects rather we're copying its history and every update produces another version of the structure while the previous versions continue to exist.
The new version reuses almost everything from the previous one and only the path that leads to the change is rebuilt and everything else is shared.This makes the cost of an update is proportional to the length of that path, not the size of the entire structure.

## The abstraction

Before reading about persistent data structures, I thought immutability was primarily about preventing accidental mutation but from litereature now I think that's just a consequence.
The deeper idea is that immutable structures separate identity from representation so that two different values can legitimately share most of their memory because neither can ever modify the shared parts.Clearly as well we see that mutation makes sharing dangerous and immutability makes sharing obvious.

## Conclusion

I started reading Okasaki expecting to learn how immutable structures copy data efficiently,I finished with a completely different understanding that immutable structures aren't efficient because they copy quickly but instead they are efficient because they avoid copying almost everything and the key takeaway is that Persistence isn't about preserving values it's about preserving structure.Once you see that, recursion will stop looking like a control-flow technique rather it becomes the mechanism that makes structural sharing possible.