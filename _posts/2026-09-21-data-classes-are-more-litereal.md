---
layout: post
author: Andrew Abok
title: Dataclasses Are More Literal Than I Expected
date: 2026-09-16 00:00:00 +0300
description: Sometimes a language feature that looks like sophisticated metaprogramming is implemented by generating ordinary source code and letting the language execute it.
categories: [Software development]
tags: [Python]
---

## Introduction

I've used Python dataclasses for years. You write a class and add `@dataclass`, and suddenly you get an __init__, __repr__, equality methods and more.I had always mentally classified this as one of those things that Python must be doing through some clever metaprogramming machinery.

Recently, while working on some PDF data extraction, I needed to compare the initial state of extracted data with the state sent back by a user after further analysis. That got me thinking about how dataclass equality was actually implemented and I finally looked at the implementation, I wasn't expecting to find quite so much string manipulation and then I saw `exec()` [here it is](https://github.com/python/cpython/blob/4eab9da960d6944546baa76e3eed56b809ea8ec0/Lib/dataclasses.py#L498).

```python
@dataclass
class User:
    name: str
    age: int
```

The point of the decorator is that we can now instantiate it like an ordinary class:

```python
user = User("Jerry",5)
print(user)
```

so from this python automatically gives you:

```python
User.__init__
User.__repr__
User.__eq__
```

Where this comes from became of interst because i was intersted to know how the equality was actually being done.
For a simple dataclass like this, the generated equality is conceptually similar to:

```python
def __eq__(self, other):
    if other.__class__ is self.__class__:
        return (
            self.name == other.name
            and self.age == other.age
        )
    return NotImplemented
```

How does a decorator turn filed definitions into executable python code?

## The surprise

I read quite a bit on data classes and ultimeately resorted to look at the implementaion in cpython, if you look in the implementaion (the link above) its looks like so:

```python
txt = f"""
def __create_fn__():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"User(name={{self.name!r}}, age={{self.age!r}})"

    return __init__, __repr__
"""

ns = {}

exec(txt, globals, ns)

init_fn, repr_fn = ns["__create_fn__"]()
```

Python isn't constructing some mysterious `__init__` implementation through a special dataclass-specific runtime mechanism. Instead, the implementation generates Python source code, executes it, retrieves the functions it created, and attaches those functions to the class..Of course, this is a simplified explanation. The actual implementation has to deal with much more than these two methods, including field ordering, defaults, keyword-only fields, frozen classes, inheritance and several other cases.

So itried replicating somethign almost similar:

```python
def generate_init(fields):
    assignments = "\n".join(
        f"        self.{name} = {name}"
        for name in fields
    )

    parameters = ", ".join(fields)

    source = f"""
def __init__(self, {parameters}):
{assignments}
"""

    namespace = {}

    exec(source, {}, namespace)

    return namespace["__init__"]
```

assining this to the user above we have :

```python
class User:
    pass

User.__init__ = generate_init(["name", "age"])

user = User("Jerry", 5)

print(user.name)
print(user.age)
```

This actually works, and if you go further to look at the source above:

```python
print(source)

def __init__(self, name, age):
        self.name = name
        self.age = age
```

The interesting part wasn't `exec()` itself. It was realizing that the function didn't need to be constructed instruction by instruction. The implementation could generate ordinary Python source code and hand that source back to Python's own execution machinery.

### Why do this?

So naturally my mind wandered on: but why would dataclasses do this?

I wouldn't pretend to know the exact motivation for why the CPython developers opted for this particular implementation. After reading through the [PEP Motivation](https://peps.python.org/pep-0557/), though, I found a few ideas that helped explain the approach, and I can summarise it like this.

At first, generating Python source like this feels unnecessarily blunt. Why not create a generic `__init__` that loops over the fields and calls `setattr`?

Partly because generated functions have real signatures. `inspect.signature(User)` can return `(name: str, age: int)`, and `help(User)` remains useful. A generic `**kwargs` loop throws a lot of that information away. Every dataclass would start to look the same to introspection tools.

Partly because the work moves from runtime to decoration time. A generic implementation would have to determine which fields exist, how they are ordered, and which have defaults when the function is called. A generated function has all of that baked into it. By the time `__init__` runs, it's essentially just doing the assignments it was generated to do.

But mostly, I think, this comes back to what dataclasses are supposed to be: syntactic sugar for a class you could have written by hand.

The implementation takes that idea surprisingly literally. Instead of building some generic object that knows how to interpret the dataclass definition at runtime, it generates something much closer to the class you would have written yourself.

The abstraction is high-level. The machinery underneath is surprisingly concrete.

### Conclusion

What surprised me wasn't that Python can execute dynamically generated code,that i was aware of, What surprised me was finding such a direct mechanism underneath a feature that feels much more sophisticated from the outside.

We tend to imagine abstractions as being backed by equally sophisticated machinery yet sometimes they're not, Sometimes the implementation is surprisingly literal:
`generate the code, compile it, execute it, attach the result`.

And perhaps that's one of the more useful reasons to read the source of the tools we use. Not because every implementation will be elegant or surprising, but because occasionally the abstraction hides something much simpler than the mental model we built for it.
