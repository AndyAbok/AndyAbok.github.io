---
layout: post
author: Andrew Abok
title: UUIDs What Are We Actually Choosing?
date: 2026-08-10 00:00:00 +0300
description: A closer look at UUIDs beyond the familiar 128-bit identifier. From how different versions are generated to what those differences actually mean for, this is an exploration of what we're really choosing when we reach for a UUID.
categories: [Software development]
tags: [C#, .NET,Python,Django]
---


## Introduction

This post was originally published on the <a href="https://blogs.innova.co.ke/what-does-your-identifier-actually-need-to-do/" target="_blank">Innova official blog</a>.

Today I'm writing about something I learned—and instead of letting it fade from memory, I'm sharing it here.

I've used UUIDs in Django and .NET for years, but recently I noticed something I had never really questioned. In Django, if I write uuid.uuid4(), the version is explicit. In ASP.NET, I usually see Guid.NewGuid() and think "UUID", without necessarily asking which version I'm getting.

That made me wonder when I choose a UUID, what am I actually choosing?

I knew UUIDs were 128 bits. I knew there were different versions. But I hadn't really internalized that the version isn't just a formatting detail, it determines how the identifier is generated and, in some cases, what information it carries.

developers often treat this:

```C#
Guid.NewGuid()
```

and this:

```python
uuid.uuid4()
```

as essentially "Give me a random ID" But underneath, you're making a decision about how that ID derives its uniqueness.

## Eight versions, eight different ideas

| Version | Core idea                   | Why you'd care                       |
| ------- | --------------------------- | ------------------------------------ |
| UUID v1 | Time + node information     | Legacy/time-ordered identity         |
| UUID v2 | DCE Security                | Specialized/rare                     |
| UUID v3 | Namespace + MD5             | Deterministic IDs                    |
| UUID v4 | Random                      | General-purpose random identifiers   |
| UUID v5 | Namespace + SHA-1           | Deterministic IDs without randomness |
| UUID v6 | Reordered v1                | Time-ordered UUIDs                   |
| UUID v7 | Unix timestamp + randomness | Modern time-ordered IDs              |
| UUID v8 | Application-defined         | Custom UUID layouts                  |

## The versions I actually care about

At this point I don't think every version deserves equal attention.

Some are interesting historically or for specialised systems, but the ones that become particularly relevant to application development are v4, v5 and v7.Take the common database case, suppose I'm creating records and decide to use `Guid.NewGuid()`
as the primary key this is convenient.

But if the generated identifiers are effectively random, then the values don't have any useful chronological relationship to one another and that raises a different question "Do I actually want my identifiers to be completely random?" Or would it be useful if the identifier also contained time information that gave me approximate chronological ordering?This is where UUID v7 becomes interesting.

Instead of treating the UUID as just an opaque 128-bit value, we can ask whether some of those bits can carry information that is useful to the system.

## Django vs ASP.NET observation

In Django:

```python
import uuid
ID = uuid.uuid4()
```

The API tells you exactly what you're asking for which is UUID version 4.
In .NET:

```C#
var ID = Guid.NewGuid()
```

you are dealing with a Guid, and the API doesn't put the UUID version in the method name.That gives you a great question:
When an API gives me a Guid, what exactly did it generate?

## Let's look at the bits

This is where things get interesting rather than just reading that a UUID has a version field, I wanted to generate UUIDs and inspect them.If UUID versions are actually encoded into the identifier, we should be able to generate one, look at its bits, and determine the version from the value itself.

```C#
var id = Guid.NewGuid();

Console.WriteLine($"UUID:    {id}");
Console.WriteLine($"Version: {GetVersion(id)}");
Console.WriteLine($"Variant: {GetVariant(id)}");

static int GetVersion(Guid guid)
{
    byte[] bytes = guid.ToByteArray();

    return (bytes[7] >> 4) & 0x0F;
}

static int GetVariant(Guid guid)
{
    byte[] bytes = guid.ToByteArray();

    return (bytes[8] >> 6) & 0x03;
}
```

On running this I wasn't just getting a Guid. I was getting a UUID with metadata on the version and stratgy.

```text
UUID:    3e51575e-eb84-4c6d-ad0b-362135a11a3c
Version: 4
Variant: 2
```

## what happens when this becomes a DB primary key?

My curosity got me to ask what happens when I use these identifiers as database keys?
Image this table

```sql
CREATE TABLE Orders
(
    SubScriptionID UUID PRIMARY KEY,
    UserID UUID NOT NULL,
    CreatedAt TIMESTAMP NOT NULL
);
```

With v4, you might generate:

```text
f91...
23a...
c72...
19b...
8df...
```

The IDs have essentially no chronological relationship.
Now image v7:

```text
0198...
0198...
0198...
0198...
0198...
```

The beginning contains timestamp information, so identifiers generated around the same time are naturally ordered much more closely

lets run this and see:

```C#
var ids = Enumerable
    .Range(1, 10)
    .Select(_ => Guid.NewGuid())
    .ToList();

Console.WriteLine("Generation order:");

foreach (var id in ids)
{
    Console.WriteLine(id);
}

Console.WriteLine();
Console.WriteLine("Sorted order:");

foreach (var id in ids.Order())
{
    Console.WriteLine(id);
}
```

```text
Generation order:
fdf1670e-9719-456d-8b5f-5680b8d6dd4d
74d8c800-9193-4265-aecb-fe5625cda718
dadfd79a-1bfd-486a-8c0c-59cf22f4dcdf
fd843d80-30d0-4340-a598-d391befe0057
2fecace2-4bc1-4c42-8dac-d68f2fa394d5
16642bc6-4ef5-4c7a-8afb-ee8189cd031c
66319c99-2645-44ac-a9d3-e85ca168dd7d
b91fe08e-202d-476b-aa3e-f0510a599b18
f6e20a23-7ea3-40d9-aba0-7580bf497014
b576b666-07b1-4f38-91f5-74d0c01e2532

Sorted order:
16642bc6-4ef5-4c7a-8afb-ee8189cd031c
2fecace2-4bc1-4c42-8dac-d68f2fa394d5
66319c99-2645-44ac-a9d3-e85ca168dd7d
74d8c800-9193-4265-aecb-fe5625cda718
b576b666-07b1-4f38-91f5-74d0c01e2532
b91fe08e-202d-476b-aa3e-f0510a599b18
dadfd79a-1bfd-486a-8c0c-59cf22f4dcdf
f6e20a23-7ea3-40d9-aba0-7580bf497014
fd843d80-30d0-4340-a598-d391befe0057
fdf1670e-9719-456d-8b5f-5680b8d6dd4d
```

You'll immediately see generation order and sorted order aren't the same and that's expected for v4.
let's try the same using  v7 (I'll ref the link to the code for brevity here).

The interesting part isn't that v7 looks different. It's that the beginning of the identifier changes with time.

```text
v4: b87d9ca2-0d60-48b3-9892-f5cb845d9211 | version 4
v7: 019feac1-622c-7f7c-9b3e-ae07c5bb1e2d | version 7

v4: bf6034f7-12b7-42f9-aa48-89871eed8566 | version 4
v7: 019feac1-622c-79e7-9c1d-5131fbab34fe | version 7

v4: 99db1ea8-3bdd-41e6-ad8c-9932195948f8 | version 4
v7: 019feac1-622c-70d3-8ce2-7dd7b78d7b3a | version 7
```

let's try this from a database property level:

```C#
var v4Ids = Enumerable
    .Range(1, 5)
    .Select(_ => Guid.NewGuid())
    .ToList();

var v7Ids = Enumerable
    .Range(1, 5)
    .Select(_ => Guid.CreateVersion7())
    .ToList();

Console.WriteLine("V4 - generation order:");

foreach (var v4id in v4Ids)
{
    Console.WriteLine(v4id);
}

Console.WriteLine();
Console.WriteLine("V4 - sorted:");

foreach (var v4idOrd in v4Ids.Order())
{
    Console.WriteLine(v4idOrd);
}

Console.WriteLine();
Console.WriteLine("V7 - generation order:");

foreach (var v7id in v7Ids)
{
    Console.WriteLine(v7id);
}

Console.WriteLine();
Console.WriteLine("V7 - sorted:");

foreach (var v7IdOrd in v7Ids.Order())
{
    Console.WriteLine(v7IdOrd);
}
```

this runs to:

```text
V4 - generation order:
04601264-308b-4073-828b-b7c68f5c9470
213372c5-1acc-4a65-8363-8a9737cdc985
8b30d5ac-c6fa-4a3e-9704-4634089bb52a
024adbd3-4f65-4cff-b7e6-c56b8f184d1e
8139785b-9ea7-468f-8105-c822c072c018

V4 - sorted:
024adbd3-4f65-4cff-b7e6-c56b8f184d1e
04601264-308b-4073-828b-b7c68f5c9470
213372c5-1acc-4a65-8363-8a9737cdc985
8139785b-9ea7-468f-8105-c822c072c018
8b30d5ac-c6fa-4a3e-9704-4634089bb52a

V7 - generation order:
019feac7-1bba-7d74-903b-af7d916af53c
019feac7-1bba-7746-8a8a-45866509186f
019feac7-1bba-745d-8bd8-13d8ef2b9d4b
019feac7-1bba-7afa-8dd6-e9b8da901c55
019feac7-1bba-7cf2-b507-c9b0c3ec82e0

V7 - sorted:
019feac7-1bba-745d-8bd8-13d8ef2b9d4b
019feac7-1bba-7746-8a8a-45866509186f
019feac7-1bba-7afa-8dd6-e9b8da901c55
019feac7-1bba-7cf2-b507-c9b0c3ec82e0
019feac7-1bba-7d74-903b-af7d916af53c
```
The values now have a strong chronological component, although they aren't necessarily strictly increasing when several UUIDs are generated within the same timestamp window.

## Does this difference actually matter?

Things will always make sense in an abstract sense the main question that kept running in my mind was something more practical "Does this difference actually matter once these identifiers become database keys?" at the end of the day the whole goal of explorartion is to see if i can use this somewhere to solve a problem.

So rather than just reading about it, I decided to test it.

### The Setup

I generated 100,000 UUID v4 values and 100,000 UUID v7 values, inserted each set into its own PostgreSQL table, and then compared the indexes and query behaviour.

The experiment isn't intended to prove that v7 is universally faster than v4. Databases, workloads, indexes and hardware all matter. What I wanted to understand was much simpler Does the ordering property of v7 give the database something useful that completely random v4 identifiers don't?

### The Tables

```sql
CREATE TABLE orders_v4
(
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE orders_v7
(
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL
);
```

```text
Inserting 100,000 UUID v4 values...
v4 insertion: 17,015 ms
Inserting 100,000 UUID v7 values...
v7 insertion: 15,538 ms
```

The v7 insertion was slightly faster, but the difference wasn't dramatic. What interested me more was the index size.

```sql
SELECT
    indexrelname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE relname IN ('orders_v4', 'orders_v7');
```

```text
orders_v4_pkey | 4384 kB
orders_v7_pkey | 3104 kB
```

The v7 index was about 30% smaller (4.3 MB vs 3.1 MB).

### Query Performance

Another thing we can look at is the query plans and how the tables are accessed.

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT *
FROM orders_v4
WHERE id = (
    SELECT id
    FROM orders_v4
    LIMIT 1
);
```

```text
"Index Scan using orders_v4_pkey on orders_v4  (cost=0.43..8.45 rows=1 width=24) (actual time=0.062..0.063 rows=1 loops=1)"
"  Index Cond: (id = $0)"
"  Buffers: shared hit=5"
"  InitPlan 1 (returns $0)"
"    ->  Limit  (cost=0.00..0.02 rows=1 width=16) (actual time=0.022..0.023 rows=1 loops=1)"
"          Buffers: shared hit=1"
"          ->  Seq Scan on orders_v4 orders_v4_1  (cost=0.00..1637.00 rows=100000 width=16) (actual time=0.021..0.021 rows=1 loops=1)"
"                Buffers: shared hit=1"
"Planning:"
"  Buffers: shared hit=19"
"Planning Time: 2.546 ms"
"Execution Time: 0.095 ms"
```

for v7

```text
"Index Scan using orders_v7_pkey on orders_v7  (cost=0.43..8.45 rows=1 width=24) (actual time=0.042..0.043 rows=1 loops=1)"
"  Index Cond: (id = $0)"
"  Buffers: shared hit=5"
"  InitPlan 1 (returns $0)"
"    ->  Limit  (cost=0.00..0.02 rows=1 width=16) (actual time=0.016..0.016 rows=1 loops=1)"
"          Buffers: shared hit=1"
"          ->  Seq Scan on orders_v7 orders_v7_1  (cost=0.00..1637.00 rows=100000 width=16) (actual time=0.015..0.015 rows=1 loops=1)"
"                Buffers: shared hit=1"
"Planning:"
"  Buffers: shared hit=11"
"Planning Time: 1.436 ms"
"Execution Time: 0.056 ms"
```

The result was actually reassuringly boring. PostgreSQL used the primary-key index for both tables. For a direct lookup by UUID, there wasn't some magical advantage to v7

That helped me separate two different questions: lookup performance and key distribution. The UUID version isn't changing the basic fact that PostgreSQL can index and look up a UUID efficiently. The more interesting difference is what happens when those keys are continuously inserted into an index.

## Conclusion

I started this by noticing that Django explicitly says `uuid.uuid4()` while .NET simply gives me `Guid.NewGuid()`. I expected the difference to be mostly about API design. Instead, it took me down into the bits of the identifier and eventually into the database.

What I came away with is that "UUID" describes the shape of the identifier, not necessarily the properties I want from it.

v4 gives me a random identifier. v7 gives me an identifier with a time component and useful ordering properties. Both are valid choices, but they are making different trade-offs.So perhaps the better question isn't "Should I use UUIDs?" but rather "What properties do I want my identifiers to have, and what happens when I put those identifiers into my database?"

You can get the code used for this write up [here](https://github.com/AndyAbok/Blog-Code/tree/main/UUID/UuidInvestigationDotnet)

## References

- [RFC 4122 - UUID Specification](https://datatracker.ietf.org/doc/html/rfc4122)
- [RFC 9562 - New UUID Formats (v6-v8)](https://datatracker.ietf.org/doc/html/rfc9562)
- [PostgreSQL UUID Type Documentation](https://www.postgresql.org/docs/current/datatype-uuid.html)
- [Microsoft GUID Documentation](https://learn.microsoft.com/en-us/dotnet/api/system.guid)