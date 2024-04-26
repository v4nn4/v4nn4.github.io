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

Imagine a simple square grid with nine dots like the one above. What if you were challenged to draw as many unique shapes or "glyphs" as possible by connecting these dots? At first glance, it seems straightforward, but let's dive deeper into the complexity.

Each of the nine dots can be connected in pairs, forming lines that are the strokes of our glyphs. Calculating all possible pairings, we find there are $(9\times8)/2=36$ unique strokes. Now, considering each stroke can either be included in or excluded from a glyph, the potential combinations explode to a staggering $2^{36} = 6.8$ billion possible glyphs! Clearly, this number is impractically high for any useful application.

To manage this complexity, we introduce a concept called equivalence classes. These classes group glyphs that are visually identical under certain transformations, such as 90° rotations and horizontal or vertical flips. This approach mirrors the challenges of traditional typography, where certain letters, like 'p', 'q', 'b', and 'd', appear as mirrored or rotated versions of each other. This set of transformations is known in mathematics as the [dihedral group](https://en.wikipedia.org/wiki/Dihedral_group) $D_4$.

{{<figure width=256 align=center caption="An example of an equivalence class. All element of the class can be transformed into each other using symmetries and rotations" src="glyph-equivalence-class.svg">}}

Next, we aim to avoid any isolated strokes. Take, for example, the French letter 'é', where the accent appears detached above the letter 'e'. To address this, we ensure that our glyphs are fully connected, meaning every stroke must link to another, creating a cohesive graph of lines without any floating elements.

By focusing only on distinct visual symbols that remain consistent under these transformations, we significantly reduce the number of relevant glyphs, making it feasible to create a unique, usable set. Let's explore how we can harness this concept to design an effective glyph generation engine.

## A search algorithm

Our objective is to identify all glyph equivalence classes. A practical approach is to first look for classes with one stroke, then two strokes, three strokes, etc. This mewthod works as a k+1-glyph is made from a k-glyph, which will be part of a k-equivalence class.

{{< figure align=center caption="All equivalence classes using 6 strokes. The 6-glyph contains all used strokes." src="6-glyph.svg" >}}

As shown in the above example, the number of classes will tend to resemble a bell curve. Initially, the number of possible classes is restricted by the need for all elements to connect, while later stages see limitations due to a diminishing number of available strokes.

{{< figure align=center caption="All equivalence classes using 10 strokes this time." src="10-glyph.svg" >}}

Here's how a basic search algorithm might operate:

1. Begin with a single stroke, considering this our initial glyph
2. For each additional stroke, check if adding it to the existing glyph forms a new valid configuration. A glyph qualifies as valid if it is connected and its equivalence class hasn't yet been identified
3. Continue this process, using existing glyphs with k strokes to explore possibilities for glyphs with k+1 strokes
4. The algorithm concludes when it creates the unique glyph that incorporates every possible stroke

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

In order to implement the `is_connected` function, we will need an adjacency matrix at stroke level. This matrix will tell if strokes intersect at least in one point and can be computed ahead of time. Then using this matrix we can build a graph linking strokes and run a [Depth-First Search](https://en.wikipedia.org/wiki/Depth-first_search) algorithm on any node to verify if all other nodes can be reached.

Checking if a glyph is part of a class is done by applying all transformations to a glyph and checking whether one of them has been already found. The transformations can be computed ahead of time for all strokes.

## Binary representation    

Since we do not need stroke coordinates at runtime, we can represent a glyph in binary as such

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

A playground is available [here](https://v4nn4.github.io/glyphs-generator/) for you to try! It is based on a Rust implementation that runs in the browser using WebAssembly and [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen).

{{<figure align=center caption="The playground lets you explore glyphs based on a set of strokes" src="app.png">}}

Python and Rust implementation repositories:
- 🐍 [v4nn4/glyphs-generator](https://github.com/v4nn4/glyphs-generator)
- 🦀 [v4nn4/glyphs-generator-rs](https://github.com/v4nn4/glyphs-generator-rs)
