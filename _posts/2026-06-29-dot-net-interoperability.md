---
layout: post
author: Andrew Abok
title: Abstraction Leakage at Language Boundaries
date: 2026-06-29 00:00:00 +0300
description:
categories: [Software development]
tags: [F#, C#, .NET]
math: true
---

## Introduction

I was working on some intergation task for one of the analytics services and the long and short of it is that there was some interop invovlved between the .Nety Eco system (C# and F#).Naturally when you hear (well, at least myself) F# and C# run on the CLR so both compile to IL what i inferred was therefore the language features should inter-operate naturally and its been working fine until when I added an optional parameter to an F# library and tried calling it from C#, I expected it to work naturally.It didn't.

That failure sent me into a rabbit hole I didn’t expect three hours of IL inspection, reflection experiments, and a growing realization that what I thought was a shared language feature wasn’t shared at all.

## The assumption that broke everything

On the surface, this should work:

```F#
namespace FSharpInterop

type FSharpUtil() =

    member _.Method(?i: int) =
        let i = defaultArg i 42
        System.Console.WriteLine(i)

```

From a C# perspective, this looks like a normal optional parameter.But when I called it in C#

```C#
var util = new FSharpUtil();
util.Method();
```

I got a complaint from the compiler

```text
error CS7036
There is no argument given that corresponds
to the required parameter 'i'
of 'FSharpUtil.Method(FSharpOption<int>)'
```

That was the first contradiction.The source says "optional".The compiler says "required".

## Peeking under the hood

So I did what i noramally do when i want to understand what happens under the hood, i decided to go and look at the lower levels whats happening I checked the IL becaause why not we are in the .Net eco system. `ilspycmd -il bin/Release/net8.0/FSharpInterop.dll`

And this is what F# actually emitted:

```IL
.method public hidebysig
        instance void Method (
            class [FSharp.Core]Microsoft.FSharp.Core.FSharpOption`1<int32> i
        ) cil managed
{
    .param [1]
        .custom instance void [FSharp.Core]
        Microsoft.FSharp.Core.OptionalArgumentAttribute::.ctor() = (
            01 00 00 00
        )

    IL_0000: ldarg.1
    IL_0001: ldc.i4.s 42
    IL_0003: call !!0 [FSharp.Core]
        Microsoft.FSharp.Core.Operators::DefaultArg<int32>(
            class [FSharp.Core]Microsoft.FSharp.Core.FSharpOption`1<!!0>,
            !!0)

    IL_0008: call void [System.Console]
        System.Console::WriteLine(int32)
    IL_000d: ret
}
```

The key realization was that the parameter is not optional at the CLR level.
It is a `FSharpOption<int>` Not an `int`.Not a `[opt] int`.A completely different representation.

## What C# does differently

When i compared this to C# my running a similar method with a default value.

```C#
public class CSharpUtil
{
    public void Method(int i = 42)
    {
        Console.WriteLine(i);
    }
}
//IL Code 
.method public hidebysig
    instance void Method (
        [opt] int32 i
    ) cil managed
{
    .param [1] = int32(42)

    IL_0001: ldarg.1
    IL_0002: call void [System.Console]
        System.Console::WriteLine(int32)
    IL_0008: ret
}
```

This is what was in the output, the important difference is not what is present.It’s what is missing.There is no runtime logic for the default, No `DefaultArg`, No branching, because C# doesn’t need it.The compiler rewrites the call site: `util.Method();` becomes `util.Method(42);`

The caller takes responsibility and the callee never knows omission happened.

## Where does the knowledge live

This is where things became interesting on the C# side the model is that (caller owns the default)

```text
Method()
→ Method(42)
→ callee receives 42
```

The information is lost before the call is made, on the F# side the model is that (callee owns the default)

```text
Method()
→ Method(None)
→ defaultArg decides
```

The information survives into the function.

## Same syntax Different architecture

That’s the part I had not internalized these are not equivalent constructs:

- `?i:int` in F#
- `int i = 42` in C#

They only look equivalent at the surface but at the IL level they are different contracts:

| Model                 | Representation                   |
| --------------------- | -------------------------------- |
| F# optional parameter | `FSharpOption<int>`              |
| C# optional parameter | `[opt] + default value metadata` |

Both models have the Same intent but very different execution model.

## Why C# cannot consume it cleanly

Going back to my main first question was this Why C# was not able to consume my optional parameter cleanly.The findings here were that C# doesn’t understand `FSharpOption<int>` as an optional parameter rather it understands it as:

```il
.param [1] = 42
[opt]
```

when C# sees F# code, it doesn’t see `optional` parameter.It sees a required `FSharpOption<int>` And the illusion breaks.

## The deeper realization: this was never a CLR feature

From all this debugging to understand why my expectaions were not met I eventually realized that Optional parameters are not a CLR abstraction.They are a compiler contract each language defines its own meaning of "optional":

- C# pushes defaulting to the caller
- F# preserves absence into the callee

The CLR just executes what it is given.This happend on another instance as well that was when i was using DU's so here are a couple of places where you are likely to face such an issue when you are doing interop between C# and F#.

| Feature              | What you think it is | What it actually is |
| -------------------- | -------------------- | ------------------- |
| F# Option            | language feature     | `FSharpOption<T>`   |
| async/await          | syntax               | state machine       |
| LINQ                 | query language       | method chains       |
| records              | immutable type       | generated class     |
| discriminated unions | algebraic type       | tagged objects      |

i cam to learn that none of these are CLR primitives and that infact they are compiler agreements.

## The abstraction leakage

The lesson i picked in this little peaking was that We often think If two languages target the same runtime, their features are compatible.But that’s not true.

They are only compatible at the level of IL instructions not at the level of language semantics.The abstraction leaks the moment two compilers disagree on:

- where defaults live
- what "missing" means
- how optionality is represented

## Conclusion

I started this investigation because an F# optional parameter wouldn’t compile from C# and i thought it was a bug, there is something i am missing on the F# side, i did a quick internet search and realized its actually an [issue](https://github.com/dotnet/fsharp/issues/5701) being fronted as a lanagauge feature.Three hours later I wasn’t thinking about optional parameters anymore.I was thinking about compiler contracts.
Many language features feel fundamental until they cross a language boundary.Then the abstraction leaks and you discover what was really there all along.Every language feature eventually becomes somebody else’s implementation detail.
