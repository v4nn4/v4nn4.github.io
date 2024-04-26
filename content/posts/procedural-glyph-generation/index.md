---
author: "v4nn4"
title: "Glyph generation"
date: "2024-04-19"
tags: ["glyph", "rust"]
ShowToc: false
ShowBreadCrumbs: false
math: katex
---

{{<figure width=256 align=center src="matrix.png">}}

Imagine you have a square grid with nine dots. How many unique shapes, or glyphs, can you create by connecting these dots with lines? Let’s calculate :
- With 9 dots we get $(9\times8)/2=36$ possible lines, or strokes, since each dot has to be linked with another one but not itself, and we want to avoid double counting
- Every stroke is either present or not in a glyph, so there are $2^{36}=6.8$ billion possible glyphs!

This is obviously too much, so let’s put some constraints on the type of glyphs that we are interested in.

First, we want to introduce the notion of equivalence classes. We define an equivalence class as a set of glyphs that are all equal in some sense. For instance, let’s say that two glyphs are equal if we can find a combination of 90° rotations, horizontal and vertical flips such that one glyph is transformed into the other one. The idea is that if a set of classes would form an alphabet, then no letter would « change » when reading upside down or through a mirror as it happens with Roman letters p, q, b and d.

{{<figure width=256 align=center caption="An example of an equivalence class. All element of the class can be transformed into each other using symmetries and rotations" src="glyph-equivalence-class.svg">}}

Then we want no hanging strokes. For instance in the French é, the accent is hanging above the letter e. We can enforce this by making sure that the glyph, viewed as a graph of strokes, is connected.

## A search algorithm

Our goal is now to find all glyph equivalence classes. A natural idea is to first look for classes with one stroke, then two strokes, three strokes, etc. This works as a k+1-glyph is made from a k-glyph, which will be part of a k-equivalence class.

{{< figure align=center caption="All equivalence classes using 6 strokes. The 6-glyph contains all used strokes." src="6-glyph.svg" >}}

As shown in the above example, the number of classes will follow a bell curve. This is expected as we are limited at the beginning because of the connectivity constraint and also at the end because fewer strokes become available.

{{< figure align=center caption="All equivalence classes using 10 strokes this time." src="10-glyph.svg" >}}

A naive search algorithm could be written as such:

1. Assume only a single 1-glyph. This will be our starting point
2. For each remaining stroke, check whether adding it to the 1-glyph creates a new valid glyph. Glyphs are valid if they are connected and if their equivalence class has not been computed yet
3. Repeat the process, using k-glyphs to search for k+1-glyphs
4. Stop with the single n-glyph that uses all n strokes

Here is an implementation in Python:

```python
def find_all_glyphs(n: int) -> dict[int, set[int]]:
    # strokes are integers from 0 to n - 1
    strokes = range(0, n)

    # Initialize 1-glyph to the first stroke
    result = {}
    result[0] = [set(0)]
    last_glyphs = result[0]

    # Look for k+1-glyphs given k-glpyhs
    for k in range(1, n):
        glyphs = []
        for last_glyph in last_glyphs:
            for stroke in strokes:
                glyph = last_glyph.add(stroke)
                if (
                    glyph not in last_glyphs
                    and glyph not in glyphs
                    and is_connected(glyph)
                    and not contains_equivalent(glyph, glyphs)
                ):
                    glyphs.append(glyph)
        result[k] = glyphs
        last_glyphs = result[k]
    
    return result
```

To check whether a glyph is connected, we will need an adjacency matrix for all strokes. This matrix will tell if strokes intersect at least in one point and can be computed ahead of time. Then using this matrix we can build a graph linking points and perform a Depth-First Search (DFS) on any node to verify if all other nodes can be reached.

Checking if a glyph is part of a class is done by applying all transformations to a glyph and checking whether one of them has been already found. The transformations can be computed ahead of time for all strokes.

Since we do not need strokes coordinates at runtime, we can represent a glyph in binary as such

$$ G = \sum_{i=1}^n w_i 2^i, \ w_i \in \\{0, 1\\} $$

This way, adding a stroke to a glyph will be written as a binary OR operation. Checking if two glyphs are equal amounts to checking if two integers are equal, which is very fast.

```python
@dataclass
class Stroke:
    index: int


@dataclass
class Glyph:
    strokes: list[Stroke]
    identifier: int

    def __eq__(self, other: "Glyph") -> bool:
        return self.identifier == other.identifier  # fast eq

    def __or__(self, other: "Glyph") -> "Glyph":
        indices = [s.index for s in self.strokes] + [s.index for s in other.strokes]
        indices = list(set(indices))
        return Glyph(
            strokes=[Stroke(index=index) for index in indices],
            identifier=self.identifier | other.identifier,
        )

    @staticmethod
    def from_stroke(stroke: Stroke) -> "Glyph":
        return Glyph(strokes=[stroke], identifier=2**stroke.index)
```

## Playground

A playground is available at https://v4nn4.github.io/glyphs-generator/ for you to try! It is based on a Rust implemetation that compiles to WebAssembly using [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen).

{{<figure align=center caption="The playground lets you explore glyphs based on a set of strokes" src="app.png">}}

Some links:
- Code repository used for this blog post : https://github.com/v4nn4/glyphs-generator
- Rust library compiled to WebAssembly : https://github.com/v4nn4/glyphs-generator-rs
