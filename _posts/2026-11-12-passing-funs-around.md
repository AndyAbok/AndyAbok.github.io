---
layout: post
author: Andrew Abok
title: Passing Functions Around in C#
date: 2026-11-12 00:00:00 +0300
description: A market-data import that paged and saved in one loop was fine for a single database. A multi-tenant background job needed the download to happen once, and a Func was how the paging loop stopped being copied.
categories: [System Design]
tags: [C#, .NET, F#]
---

Welcome to another write up of something i learnt,So genrally I was already comfortable passing functions around coming from
functional programming, and having spent time working in F#, it was
normal to pass a function to another function and let it decide when to call it. I had used `System.Func` before too, when calling a C# library from F#. What I had not done as often was reach for it naturally in C#.

A couple of weeks ago while I was working on a market-data import in .NET i ran into a scenario that made me learn that Funcs are just delegates in C#.

I already knew what a slow import felt like. I wrote about one in When Write hurt more than reads,
where the pain was the database and the fix was to stop saving one row
at a time. So when working on a data import service in .NET, I
reached for the same shape.Page through the API then map each price. Buffer a few hundred. Write the buffer. If a batch fails, retry that batch one row at a time so one bad record does not throw away the rest, well this is more than that but you get my drift i was bowwrong alot of my thought process from there.

That service worked. It worked because there was one database on the other side of it.

## The original import methods

I had two methods that were central to the market-data import SynchronizeAsync and ImportAsync.

`SynchronizeAsync` and `ImportAsync` were the methods the controllers already called. One walked forward from the last sync date. The other took a range, which is also what a retry wants when a failed window has to be pulled again. Under both of them sat the same loop:

```csharp
var pageNumber = 1;
var buffer = new List<StockPrice>(SaveBatchSize);

while (true)
{
    var response = await client.GetPricesAsync(
        apiKey, exchange, start, end, pageNumber, PageSize, cancellationToken);

    if (response.Data is null || response.Data.Count == 0)
        break;

    await BatchAndSaveAsync(response.Data, buffer, summary, cancellationToken);

    if (pageNumber >= response.TotalPages)
        break;

    pageNumber++;
}

await SaveBatchAsync(buffer, summary, cancellationToken);
```

What i did was first fetch, map, save, next page. For one tenant that is an honest loop. Existing objs were looked up once per batch, not once per row. I was not worried about this code.

## Then there was more than one database

The background job was not importing into one database. Each tenant has its own connection string. If the job kept the loop above, every tenant would call the market API for the same prices, then write them into its own database.

Fetch and save were glued together, so the only way to save for a tenant was to fetch again.

After talking it through with one of the engineering team, the split was the part I had missed:

1. Download the prices once.
2. Hand that batch to each tenant.
3. Let each tenant map and save on its own connection, in parallel.

The controller methods stayed. A manual import for one tenant still wants to page and save together, and those methods were already exposed. The background job needed a path beside them, not a rewrite of the first.

The new methods only download:

```csharp
public Task<MarketDataBatch> FetchEquitesDataSinceLastAsync(
    DateTime previousSyncDate,
    CancellationToken cancellationToken = default)
{
    return FetchAllAsync(
        (page, token) => _marketStatsClient.GetPricesFromLastRequestAsync(
            apiKey, exchange, previousSyncDate, page, PageSize, token),
        previousSyncDate,
        DateTime.Now,
        cancellationToken);
}
```

`FetchEquitesDataRangeAsync` is the same idea with an explicit start and end.

The worker does not fetch the data rather it receives the batch:

```csharp
await Parallel.ForEachAsync(tenants, parallelOptions, async (tenant, token) =>
{
    await ImportForTenantAsync(tenant, batch, token);
});
```

Inside one tenant, nothing is shared with the next tenant given `DbContext` is not thread-safe.A fresh context per tenant is the safety story. The downloaded list is only read. Each save owns its own change tracker, its own transaction, and its own failure count.

## The loop was copied four times

Once fetch and save were allowed to be different methods, I could see the paging loop in four places. `ImportNewDataAsync`. `ImportRangeAsync`. And the two new fetch methods. The body barely changed what changed was which client method ran on each page, and what we did with the page after it arrived.

I knew this could be one loop it felt redundant and i was very uncomfartble having it in the code base. When a `Func` showed up as the way to pass "which call to make", I stalled on it first i went with what worked which was the repetetive loop.

## I had already used Func, in F#

The first time I met `System.Func` it was not in C#. I was fitting a Nelson–Siegel–Svensson curve from F#, and MathNet's optimizer is a C# API. It does not take an F# function. It takes a delegate:

```fsharp
let convertedFunction = new System.Func<Vector<float>, float>(Vector.toList >> f)
```

 Basically from the inside this reads as `f` is the function I wrote, the one that scores a parameter vector. `Vector.toList >> f` composes two functions: turn MathNet's vector into an F# list, then hand that list to `f`. `new System.Func<_, _>(...)` wraps that F# function as a CLR delegate, so C# can call it.

A `Func` was never a C# trick but i came to learn that it is a method with a type. F# makes passing functions so ordinary that I forgot the type was there. Seeing the same type in a paging loop, I treated it like something I had not earned yet.

## What Func is in C#

`Func<T1, T2, TResult>` is a generic delegate in `System`. Every type argument except the last one is an input. The last one is what the call returns. There is a form for zero arguments through sixteen. If the method returns nothing, the type is `Action` instead. Same idea just no result.

This signature:

```csharp
Func<int, CancellationToken, Task<PagedResult<List<MarketPrice>>>>
```

is only a shape:

- pass an `int`, the page number
- pass a `CancellationToken`
- get back a `Task` of one page

The important thing i picked when digging through the docs here is that the delegate describes the shape of the
call. It does not need to know where the values come from or what the
implementation does internally.

You can create a compatible delegate in a few ways: with a lambda, a
local function, or a method group.

A lambda:

``` csharp
Func<int, CancellationToken, Task<PagedResult<List<MarketPrice>>>> fetchPage =
    (page, token) => client.GetPricesAsync(
        key, exchange, start, end, page, pageSize, token);
```

A local function:

``` csharp
Task<PagedResult<List<MarketPrice>>> FetchPage(
    int page, CancellationToken token) =>
    client.GetPricesFromLastRequestAsync(
        key, exchange, since, page, pageSize, token);

Func<int, CancellationToken, Task<PagedResult<List<MarketPrice>>>> fetchPage =
    FetchPage;
```

And a method group, where an existing method already has a compatible
signature:

``` csharp
Func<int, CancellationToken, Task<PagedResult<List<MarketPrice>>>> fetchPage =
    service.FetchPageAsync;
```

The lambda is allowed to close over the things that differ between call sites: the dates, the key, which client method. The loop does not know any of that. It only knows how to ask for the next page.

That is the whole paging method. One function gets the page. Another function decides what to do with it.

```csharp
private async Task PageThroughAsync(
    Func<int, CancellationToken, Task<PagedResult<List<MarketPrice>>>> fetchPage,
    Func<IReadOnlyList<MarketPrice>, CancellationToken, Task> onPage,
    CancellationToken cancellationToken)
{
    var pageNumber = 1;

    while (true)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var response = await fetchPage(pageNumber, cancellationToken);

        if (response.Data is null || response.Data.Count == 0)
            break;

        await onPage(response.Data, cancellationToken);

        if (pageNumber >= response.TotalPages)
            break;

        pageNumber++;
    }
}
```

The background download accumulates. `onPage` appends, and because that work is synchronous we return an already completed task so the signature still matches:

```csharp
var fetched = new List<MarketPrice>();

await PageThroughAsync(
    fetchPage,
    (page, _) =>
    {
        fetched.AddRange(page);
        return Task.CompletedTask;
    },
    cancellationToken);
```

The controller import does not want the whole history in memory. Its `onPage` is the batch save, which already returns a `Task`, so it fits the same slot with no wrapper:

```csharp
await PageThroughAsync(
    fetchPage,
    (page, token) => BatchAndSaveAsync(page, buffer, summary, token),
    cancellationToken);

await SaveBatchAsync(buffer, summary, cancellationToken);
```

`ImportRangeAsync` passes `GetPricesAsync`. The sync path passes `GetPricesFromLastRequestAsync`. Same loop. The function arguments are the only moving parts.

Nothing here is lazy, and nothing is running in parallel yet. `Func` does not schedule work. It stores a method so something else can call it later. `await fetchPage(...)` is an ordinary call.

## What i Learnt

1. **Do not fuse a shared download to a per-tenant save.** The moment there is more than one database, fetch and persist are different jobs.

2. **Parallel work needs its own context.** Share the batch not the `DbContext`.

3. **`Func` is a typed method you can pass.** If two loops differ only by the call in the middle, that call is the argument.

The service, the paging loop, and the per-tenant worker are in the [Blog-Code repo](https://github.com/AndyAbok/Blog-Code/tree/main/MarketDataImport).

## Conclusion

The Django import taught me to write less often. This one taught me to
fetch less often, and then to notice that the thing I was copying was
not the save. It was the loop.

The loop only needed a function.
