# Zebra
A vector database for querying meaningfully similar data.

## Features
* **On-disk database index**, minimising memory impact for large datasets
* Distance metrics and embedding models are provided, though external implementations can be supplied
* Inserting, **deleting**, and querying vectors
* **[No memory-mapped (MMAP) file IO](https://db.cs.cmu.edu/mmap-cidr2022/)**
* Parallelised database operations; multithreaded reads & writes are safe

### Motivation
Approximate nearest neighbour search for finding semantically-similar documents is a common use case, and a variety of existing solutions exist that are often described as 'vector databases'. Many of these solutions are offered as services, and some exist as libraries.

While designing a content recommendation system for [Oku](https://okubrowser.github.io), several requirements became clear:
* Safe multithreaded database access
* Inserting, deleting, and querying records
* On-disk storage of database contents
* Support for multiple [modalities](https://en.wikipedia.org/wiki/Modality_(semiotics)) of information

Oku enables distributed storage & distribution of mutable user-generated data; therefore, any embedded database used for semantic search needed to be:
* Capable of asynchronous read & write access
* Capable of creating, reading, updating, and deleting (**[CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete)**) records **without downtime or excessive resource usage**
* Capable of scaling in dataset size **without excessively impacting memory usage for an individual node**
* Capable of acceptable recall with multiple distance metrics

Despite the common need—a scalable CRUD database—existing solutions often fell short.

#### Distribution & CRUD
Many vector databases utilise the [hierarchical navigable small world (HNSW)](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) algorithm to construct their database indices, as it (a) achieves high recall on high-dimensional data regardless of distance metric, and (b) performs fast queries regardless of dataset size. However, despite its attractiveness on benchmarks, it can be impractical to use in many production contexts as (a) it's difficult to distribute as you cannot shard the index across multiple nodes, (b) the entire index must be loaded into memory to perform operations, making memory a bottleneck in addition to storage, and (c) deleting vectors essentially requires rebuilding the entire index from scratch and re-inserting every vector except the deleted ones; using redundant indices and tombstoning is the only way to keep the database online.

The need for a scalable & mutable vector database is not new, however, and the problem has apparently been solved to an acceptable degree before—content recommendation systems based on embedding vectors have been in production for many years, and they've often used some variation of [locality sensitive hashing (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) to build a vector database index. An LSH index is not graph-based, but instead breaks up an *f*-dimensional space into regions of similar vectors. Consequently, it can be (a) sharded, (b) accessed in parallel, and (c) accessed from storage because, unlike a graph such as HNSW, it is not '[object soup](https://jacko.io/object_soup.html)' and avoids issues with cache locality and synchronisation in multithreaded contexts. LSH's advantages in performance and resource usage does come with an implication: while HNSW approximates neighbours, LSH approximates similarities. The recall of LSH is lesser as it's less concerned with finding *the nearest* neighbours, and more concerned with just finding what *is near*. For fine-grained searches, LSH is less helpful, but for large & varied datasets where it's important to find records that are 'close enough', it has significant advantages.

#### Integrity & Safety
To avoid excessive memory usage, some have saved indexes to storage and performed operations directly on the index files as if they were in memory, taking advantage of a technique called [memory mapping (`mmap`)](https://en.wikipedia.org/wiki/Memory-mapped_file). Spotify boasts of [its LSH index](https://github.com/spotify/annoy):
> … you can share index across processes … you can pass around indexes as files and map them into memory quickly … You can also pass around and distribute static files to use in production environment, in Hadoop jobs, etc. Any process will be able to load (mmap) the index into memory and will be able to do lookups immediately.

Cloudflare [makes similar bold claims](https://blog.cloudflare.com/scalable-machine-learning-at-cloudflare/) regarding its use of `mmap`:
> Leveraging the benefits of memory-mapped files, wait-free synchronization and zero-copy deserialization, we've crafted a unique and powerful tool for managing high-performance, concurrent data access between processes.
>
> The data is stored in shared mapped memory, which allows the `Synchronizer` to “write” to it and “read” from it concurrently. This design makes `mmap-sync` a highly efficient and flexible tool for managing shared, concurrent data access.
>
> In the wake of our redesign, we've constructed a powerful and efficient system that truly embodies the essence of 'bliss'. Harnessing the advantages of memory-mapped files, wait-free synchronization, allocation-free operations, and zero-copy deserialization, we've established a robust infrastructure that maintains peak performance while achieving remarkable reductions in latency.

Multiprocess concurrency with `mmap` is arguably *impossible*—countless DBMSes have learned the same lesson after many years. There is [a paper on this subject](https://db.cs.cmu.edu/papers/2022/cidr2022-p13-crotty.pdf) that covers the pitfalls of `mmap` in detail; suffice it to say, a memory-mapped database index has not demonstrably achieved the data integrity & memory-safety guarantees necessary for a production database.

#### Potential Improvements
This software is free & open-source (FOSS), and code contributions are welcome.
Its usage within Oku involves operating on consumer hardware with data that is diverse. For use-cases where more hardware resources are available or datasets needing greater recall are used, this database could be extended with a new implementation of the HNSW algorithm, providing a mutable, multithreaded, real-time variant of HNSW. Such an index may be possible, as [Vespa claims to have created such an implementation](https://docs.vespa.ai/en/approximate-nn-hnsw.html). Improving HNSW's memory usage, however, appears impossible due to its fundamental graph-based nature.

##### The name
Zebras make a 'neigh' sound. The database performs an [approximate nearest-*neigh*bour search](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods) to find similar data.

## Installation
Zebra is intended for use as an embedded database. You can add it as a dependency to a Rust project with the following command:
```sh
cargo add --git "https://github.com/emmyoh/zebra"
```

Additionally, a command-line interface (CLI) exists for basic usage in the terminal.

With the [Rust toolchain](https://rustup.rs/) installed, run the following command:
```sh
cargo install --git https://github.com/emmyoh/zebra --features="cli"
```

You should specify the features relevant to your use case. For example, if you're interested in using the Zebra CLI on an Apple silicon device, run:
```sh
cargo install --git https://github.com/emmyoh/zebra --features="cli,accelerate,metal"
```

### Features
* `default_db` - Provides default configurations for databases.
* `accelerate` - Uses Apple's Accelerate framework when running on Apple operating systems.
* `cuda` - Enables GPU support with Nvidia cards.
* `mkl` - Uses Intel oneMKL with Intel CPUs and GPUs.
* `metal` - Enables GPU support for Apple silicon machines.
* `sixel` - Prints images in Sixel format when using the CLI with compatible terminals.
* `cli` - Provides a command-line interface to Zebra.