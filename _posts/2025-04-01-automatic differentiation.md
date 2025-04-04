---
layout: post
title: AUTOMATIC DIFFERENTIAON (Part 2)
date: 2025-04-01 00:00:00 +0300
categories: [Optimization]
tags: [Optimization,Machine Learning,F#]
canonical_url: https://blogs.innova.co.ke/automatic-differentiation/
math: true
---

This post was originally published on the <a href="https://blogs.innova.co.ke/automatic-differentiation/" target="_blank">Innova official blog</a>.

Last time we explored Automatic differentiation we looked at <a href="https://andyabok.github.io/posts/automatic-differentiation/" target="_blank">forward mode Automatic Differentiation</a>.The derivatives were computed by traversing the computational graph from inputs to outputs.But today we flip the script literally. 

**Reverse mode AD** is like solving a mystery backward: instead of asking *"How does this input affect the output?"* we ask *"Which outputs does this influence?"* This shift in perspective makes it efficient,especially when dealing with functions with many inputs and few outputs. 

## The Two Acts of Reverse Mode
The reverse mode consists of two passes through the graph.

### Act 1:The Forward pass - Building the scene
we start by constructing the computational graph,just as before, but this time we're only storing partial derivatives at each node not computing full gradients yet.Think of it as laying all the clues before solving the case 

I'll borrow this common example you'll find in texts: 
$$
z = x.y + sin(x)
$$

The computation l graph breaks down into:
1. Inputs: $x,y$
2. Intermediate steps:
    * $a = x . y$
    * $b = sin(x)$
3. Output : $z = a + b$

### Act 2 : The backward Pass - Unraveling the chain

Now, the real magic happens.We start from the output $z$ and work backward,applying the chain rule at each step.

To do this, we introduce adjoint variables $\bar{x_i}$ representing how much each variable contributes to the final output.

1. At the outut $\bar{z} = \frac{\partial{z}}{\partial{z}} = 1$

2. Moving backward, we compute how $a$ and $b$ influence $z$:
    * $\bar{a} = \bar{z} . \frac{\partial{z}}{\partial{a}} = \bar{z} . 1$ 
    * $\bar{b} = \bar{z} . \frac{\partial{z}}{\partial{b}} = \bar{z} . 1$ 

3. Now, how does y affect $a$
    * $\bar{y} = \bar{a} . \frac{\partial{a}}{\partial{y}} = \bar{a} . x$ 

4. So given $x$ influences both $a$ and $b$, so we sum its contributions:
    * $\bar{x} = \bar{a} . \frac{\partial{a}}{\partial{a}} = \bar{b} .\frac{\partial{b}}{\partial{x}} =   \bar{a} . y + \bar{b} . cos(x)$ 
   
This sums up the essence of reverse accumulation.Instead of propagating derivatives forward, we gather them backward, stitching together the full gradient piece by piece.        

we can translate this derivatives into pseudocode we will replace the derviatives with adjoint variables.

> gz = ?
> 
> gb = gz
> 
> ga = gz
> 
> gy = x * ga
> 
> gx = y * ga + cos(x) * gb


## Strengths of Reverse Mode 
* For a function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ like a losss function all the partial derivatives are computed in one sweep while forward mode will need $n$ passes.

* Deep learning's Best friend Neural networks often have millions of inputs (parameters) and just one output(loss) reverse mode is tailor made for this. 

## F# Implementation 

In this section, we'll build the reverse mode AD in F# and leverage the strengths of F#.

### Expression Language

We will begin by defining the mathematical operations as a discriminated union.

```fsharp
//operations
type Expr =
    | Var of string
    | Const of float
    | Add of Expr * Expr
    | Mul of Expr * Expr
    | Exp of Expr
    | Cos of Expr
    | Sin of Expr
    | Square of Expr
    | TimesTwo of Expr
```
### Building the Graph 
we convert the `Expr` into a computation graph (GraphNode) that tracks values and derivatives

```fsharp
type GraphNode = {
    mutable Value: float
    mutable Adjoint: float
    Children: GraphNode list
    Operation: Operation
}
```

### Forward Pass 
Compute the function value by evaluating the nodes in order.

```fsharp
let forwardPass (env: Map<string, float>) (root: GraphNode) (expr: Expr) =
    let rec eval node expr =
        match expr with
        | Var name -> node.Value <- env[name]
        | Const value -> node.Value <- value
        | Add (a, b) ->
            let aNode = node.Children[0]
            let bNode = node.Children[1]
            eval aNode a
            eval bNode b
            node.Value <- aNode.Value + bNode.Value
        | Mul (a, b) ->
            let aNode = node.Children[0]
            let bNode = node.Children[1]
            eval aNode a
            eval bNode b
            node.Value <- aNode.Value * bNode.Value
        | Exp e ->
            let eNode = node.Children[0]
            eval eNode e
            node.Value <- Math.Exp(eNode.Value)
        | Cos e ->
            let eNode = node.Children[0]
            eval eNode e
            node.Value <- Math.Cos(eNode.Value)
        | Sin e ->  
            let eNode = node.Children[0]
            eval eNode e
            node.Value <- Math.Sin(eNode.Value)
        | Square e ->
            let eNode = node.Children[0]
            eval eNode e
            node.Value <- eNode.Value * eNode.Value
        | TimesTwo e ->
            let eNode = node.Children[0]
            eval eNode e
            node.Value <- 2.0 * eNode.Value
    
    eval root expr
    root.Value
```

### Backward Pass 
Here the gradient is propagated using the chain rule.

```fsharp
let backwardPass (root: GraphNode) =
    root.Adjoint <- 1.0
    
    let rec propagate (node: GraphNode) =
        match node.Operation with
        | AddOp ->
            let a, b = node.Children[0], node.Children[1]
            a.Adjoint <- a.Adjoint + node.Adjoint
            b.Adjoint <- b.Adjoint + node.Adjoint
            propagate a
            propagate b
        | MulOp ->
            let a, b = node.Children[0], node.Children[1]
            a.Adjoint <- a.Adjoint + node.Adjoint * b.Value
            b.Adjoint <- b.Adjoint + node.Adjoint * a.Value
            propagate a
            propagate b
        | ExpOp ->
            let e = node.Children[0]
            e.Adjoint <- e.Adjoint + node.Adjoint * Math.Exp(e.Value)
            propagate e
        | CosOp ->
            let e = node.Children[0]
            e.Adjoint <- e.Adjoint + node.Adjoint * -Math.Sin(e.Value)
            propagate e
        | SinOp -> 
            let e = node.Children[0]
            e.Adjoint <- e.Adjoint + node.Adjoint * Math.Cos(e.Value)
            propagate e
        | SquareOp ->
            let e = node.Children[0]
            e.Adjoint <- e.Adjoint + node.Adjoint * 2.0 * e.Value
            propagate e
        | TimesTwoOp ->
            let e = node.Children[0]
            e.Adjoint <- e.Adjoint + node.Adjoint * 2.0
            propagate e
        | Leaf -> ()
    
    propagate root
```


Let's compute $f(x1, x2) = cos(x2) + exp(2*x1) + x1*x2^2$:

```fsharp
//lets  def an expression: cos(x2) + (exp(2*x1) + x1*x2²) + 0
let testExpr = 
    Add(
        Add(
            Cos(Var "x2"),
            Add(
                Exp(TimesTwo(Var "x1")),
                Mul(Var "x1", Square(Var "x2"))
            )
        ),
        Const 0.0
    )

let env = Map ["x1", 1.0; "x2", 2.0]

//Build the graph
let root, cache = buildGraph testExpr Map.empty

//function value forward pass
forwardPass env root testExpr |> ignore
backwardPass root
let x1Node = cache[Var "x1"]
let x2Node = cache[Var "x2"]
```

### Conclusions. 
The purpose of this write up was to understand reverse mode AD and how it works and using F# in doing so making this implementation not as robust diffsharp\Furnace does the work of autodiff in F# properly.

