"""3Blue1Brown-style explainer clips for the class deck.

    manim -r 1920,1080 --fps 30 docs/slides/anim/scenes.py ThreadsAndWarps
"""

import math
import random

from manim import (
    BLUE_B, BLUE_D, DOWN, GREY_B, GREY_D, LEFT, ORIGIN, PI, RED_C, RIGHT, TEAL_C, UP, WHITE, YELLOW,
    AnimationGroup, ArcBetweenPoints, Arrow, Circle, Create, FadeIn, FadeOut, GrowFromEdge, LaggedStart, Line,
    Rectangle, ReplacementTransform, Scene, Square, SurroundingRectangle, Text, Transform, VGroup, Write,
)

SERIF = "Palatino"
MONO = "Menlo"


def heading(text):
    return Text(text, font=SERIF, font_size=40).to_edge(UP, buff=0.5)


def caption(text, size=28):
    return Text(text, font=SERIF, font_size=size, color=GREY_B).to_edge(DOWN, buff=0.6)


def code(text, size=24):
    return Text(text, font=MONO, font_size=size, color=YELLOW)


class ThreadsAndWarps(Scene):
    def construct(self):
        title = heading("One query head is 128 numbers")
        cells = VGroup(*[
            Square(0.09, stroke_width=0, fill_opacity=1, fill_color=BLUE_D if (i // 4) % 2 == 0 else BLUE_B)
            for i in range(128)
        ]).arrange(RIGHT, buff=0.012).shift(UP * 1.2)
        self.play(Write(title))
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.01, run_time=1.5))
        self.wait(0.5)

        groups = [VGroup(*cells[4 * l: 4 * l + 4]) for l in range(32)]
        lanes = VGroup(*[
            Circle(0.12, color=WHITE, stroke_width=2).move_to([g.get_center()[0], -0.6, 0]) for g in groups
        ])
        links = VGroup(*[Line(g.get_bottom(), d.get_top(), stroke_width=1, color=GREY_B) for g, d in zip(groups, lanes)])
        self.play(Transform(title, heading("A thread, or lane, owns 4 of them")))
        self.play(LaggedStart(*[Create(x) for x in lanes], lag_ratio=0.03), LaggedStart(*[Create(x) for x in links], lag_ratio=0.03))
        lane0 = Text("lane 0", font=SERIF, font_size=22).next_to(lanes[0], DOWN, buff=0.2)
        lane31 = Text("lane 31", font=SERIF, font_size=22).next_to(lanes[31], DOWN, buff=0.2)
        self.play(FadeIn(lane0), FadeIn(lane31))
        box0 = SurroundingRectangle(VGroup(groups[0], lanes[0]), color=YELLOW, buff=0.08)
        note0 = Text("columns 0 to 3", font=SERIF, font_size=24, color=YELLOW).next_to(box0, UP, buff=0.15).shift(RIGHT * 0.9)
        self.play(Create(box0), FadeIn(note0))
        self.wait(1)
        self.play(FadeOut(box0), FadeOut(note0))

        warp_box = SurroundingRectangle(lanes, color=TEAL_C, buff=0.15, corner_radius=0.1)
        warp_label = Text("warp: 32 lanes that run each instruction together", font=SERIF, font_size=28, color=TEAL_C)
        warp_label.next_to(warp_box, DOWN, buff=0.6)
        self.play(Transform(title, heading("32 lanes make a warp")), Create(warp_box), FadeOut(lane0), FadeOut(lane31))
        self.play(Write(warp_label))
        self.wait(0.5)

        for instr in ["k = load 4 numbers of K", "partial = q · k"]:
            line = code(instr).move_to(UP * 2.3)
            self.play(FadeIn(line, shift=DOWN * 0.2), run_time=0.5)
            self.play(AnimationGroup(*[d.animate.set_fill(YELLOW, opacity=1) for d in lanes]), run_time=0.4)
            self.play(AnimationGroup(*[d.animate.set_fill(YELLOW, opacity=0) for d in lanes]), run_time=0.4)
            self.play(FadeOut(line), run_time=0.3)

        end = caption("Each lane now holds a partial dot product. 32 partials must become one score.")
        self.play(FadeIn(end))
        self.wait(2.5)


class WarpShuffle(Scene):
    def construct(self):
        random.seed(3)
        values = [random.randint(0, 9) for _ in range(32)]
        spacing = 0.42
        xs = [(i - 15.5) * spacing for i in range(32)]
        title = heading("Combining 32 partials with xor shuffles")
        dots = VGroup(*[Circle(0.14, color=WHITE, stroke_width=2).move_to([x, -1.6, 0]) for x in xs])
        nums = VGroup(*[Text(str(v), font=MONO, font_size=18).move_to([x, -1.05, 0]) for v, x in zip(values, xs)])
        self.play(Write(title))
        self.play(LaggedStart(*[Create(d) for d in dots], lag_ratio=0.02), LaggedStart(*[FadeIn(n) for n in nums], lag_ratio=0.02))
        self.wait(0.5)

        for rnd, dist in enumerate([16, 8, 4, 2, 1], start=1):
            call = code(f"s += __shfl_xor_sync(0xffffffff, s, {dist})").move_to(UP * 2.2)
            label = Text(f"round {rnd} of 5: lane i trades with lane i xor {dist}", font=SERIF, font_size=26, color=GREY_B)
            label.next_to(call, DOWN, buff=0.25)
            arcs = VGroup(*[
                ArcBetweenPoints(dots[i].get_bottom() + DOWN * 0.05, dots[i ^ dist].get_bottom() + DOWN * 0.05,
                                 angle=PI / 2 if dist > 2 else PI * 0.8, stroke_width=1.5, color=TEAL_C)
                for i in range(32) if i < (i ^ dist)
            ])
            self.play(FadeIn(call), FadeIn(label), run_time=0.5)
            self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.03), run_time=1.0)
            values = [values[i] + values[i ^ dist] for i in range(32)]
            new_nums = VGroup(*[Text(str(v), font=MONO, font_size=18 if v < 100 else 13).move_to([x, -1.05, 0]) for v, x in zip(values, xs)])
            self.play(Transform(nums, new_nums), run_time=0.7)
            self.play(FadeOut(arcs), FadeOut(call), FadeOut(label), run_time=0.4)

        box = SurroundingRectangle(nums, color=YELLOW, buff=0.12)
        self.play(Create(box))
        self.play(FadeIn(caption("5 rounds, no shared memory: every lane holds the full score.")))
        self.wait(2.5)


class GridBlocksSMs(Scene):
    def construct(self):
        title = heading("Thread, warp, block, grid")
        self.play(Write(title))

        thread = Circle(0.14, color=WHITE, stroke_width=2).move_to(LEFT * 5.2 + UP * 0.8)
        t_label = Text("thread", font=SERIF, font_size=24).next_to(thread, DOWN, buff=0.3)
        self.play(Create(thread), FadeIn(t_label))

        warp_dots = VGroup(*[Circle(0.05, color=WHITE, stroke_width=1.5) for _ in range(32)]).arrange(RIGHT, buff=0.03)
        warp_dots.move_to(LEFT * 2.2 + UP * 0.8)
        warp_box = SurroundingRectangle(warp_dots, color=TEAL_C, buff=0.08)
        w_label = Text("warp: 32 threads", font=SERIF, font_size=24).next_to(warp_box, DOWN, buff=0.3)
        self.play(ReplacementTransform(thread.copy(), warp_dots), Create(warp_box), FadeIn(w_label))

        block = Square(0.9, color=BLUE_B, fill_opacity=0.25).move_to(RIGHT * 1.6 + UP * 0.8)
        b_label = Text("block: one warp here.\nRuns on one SM.", font=SERIF, font_size=24, line_spacing=0.8).next_to(block, DOWN, buff=0.3)
        self.play(ReplacementTransform(VGroup(warp_dots, warp_box).copy(), block), FadeIn(b_label))

        grid = VGroup(*[Square(0.32, color=BLUE_B, fill_opacity=0.6, stroke_width=1) for _ in range(16)]).arrange_in_grid(4, 4, buff=0.06)
        grid.move_to(RIGHT * 4.9 + UP * 0.8)
        g_label = Text("grid: one block per\nquery head. 16 at batch 1.", font=SERIF, font_size=24, line_spacing=0.8).next_to(grid, DOWN, buff=0.3)
        self.play(ReplacementTransform(block.copy(), grid), FadeIn(g_label))
        self.wait(1.5)

        everything = VGroup(thread, t_label, warp_dots, warp_box, w_label, block, b_label, g_label)
        self.play(FadeOut(everything), Transform(title, heading("The RTX 3090 has 82 SMs")), grid.animate.move_to(LEFT * 5 + DOWN * 0.2))

        sms = VGroup(*[Square(0.42, color=GREY_D, stroke_width=1.5) for _ in range(82)]).arrange_in_grid(6, 14, buff=0.08)
        sms.move_to(RIGHT * 1.6 + DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(s) for s in sms], lag_ratio=0.01, run_time=1))
        self.play(Transform(title, heading("Path A at batch 1")))
        moved = VGroup(*[Square(0.42, color=RED_C, fill_opacity=0.85, stroke_width=1.5).move_to(sms[i]) for i in range(16)])
        self.play(ReplacementTransform(grid, moved), run_time=1.5)
        cap = caption("16 of 82 SMs work. The other 66 sit idle.")
        self.play(FadeIn(cap))
        self.wait(2)

        self.play(Transform(title, heading("Path B: split each head's history into chunks")), FadeOut(cap))
        chunks = VGroup(*[
            Square(0.42, color=TEAL_C, fill_opacity=0.85, stroke_width=1.5).move_to(sms[i]) for i in range(80)
        ])
        sources = VGroup(*[moved[i % 16].copy() for i in range(80)])
        self.play(FadeOut(moved), ReplacementTransform(sources, chunks), run_time=2)
        cap2 = caption("16 heads × 5 chunks = 80 blocks. Nearly every SM works, then the chunks merge.")
        self.play(FadeIn(cap2))
        self.wait(2.5)


def online(scores):
    m, l, trace = -math.inf, 0.0, []
    for z in scores:
        new_m = max(m, z)
        l = l * math.exp(m - new_m) + math.exp(z - new_m)
        m = new_m
        trace.append((m, l))
    return trace


def fmt(x):
    return "−∞" if x == -math.inf else f"{x:.2f}"


class OnlineSoftmax(Scene):
    def construct(self):
        scores = [0.4, 1.3, -0.2, 2.1, 0.9, 1.7, -0.8, 2.6, 1.1, 0.2, 1.9, 0.5]
        title = heading("Online softmax: one pass, a running state")
        self.play(Write(title))

        tokens = VGroup(*[Square(0.75, color=BLUE_B, stroke_width=2) for _ in scores]).arrange(RIGHT, buff=0.12).shift(UP * 1.1)
        labels = VGroup(*[Text(f"{z:.1f}", font=MONO, font_size=20).move_to(t) for z, t in zip(scores, tokens)])
        tag = Text("score of each token, q · k", font=SERIF, font_size=24, color=GREY_B).next_to(tokens, UP, buff=0.25)
        self.play(LaggedStart(*[Create(t) for t in tokens], lag_ratio=0.05), FadeIn(labels), FadeIn(tag))

        def state_text(m, l):
            return VGroup(
                Text(f"m = {fmt(m)}", font=MONO, font_size=30),
                Text(f"l = {fmt(l) if l else '0.00'}", font=MONO, font_size=30),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(DOWN * 1.3)

        rules = Text("m: largest score so far    l: sum of e^(z − m) so far", font=SERIF, font_size=24, color=GREY_B)
        rules.move_to(DOWN * 2.8)
        state = state_text(-math.inf, 0)
        self.play(FadeIn(state), FadeIn(rules))
        pointer = Arrow(UP * 0.1, DOWN * 0.5, color=YELLOW, buff=0).next_to(tokens[0], DOWN, buff=0.1)
        self.play(FadeIn(pointer))
        for i, (m, l) in enumerate(online(scores)):
            self.play(pointer.animate.next_to(tokens[i], DOWN, buff=0.1), tokens[i].animate.set_fill(YELLOW, opacity=0.25), run_time=0.3)
            self.play(Transform(state, state_text(m, l)), run_time=0.35)
        full_m, full_l = online(scores)[-1]
        self.wait(1)

        self.play(Transform(title, heading("Path B: three chunks, then a merge")), FadeOut(pointer), FadeOut(rules), FadeOut(state), FadeOut(tag))
        colors = [BLUE_D, TEAL_C, RED_C]
        parts = []
        for c in range(3):
            seg = scores[4 * c: 4 * c + 4]
            m, l = online(seg)[-1]
            parts.append((m, l))
            self.play(*[tokens[i].animate.set_fill(colors[c], opacity=0.5) for i in range(4 * c, 4 * c + 4)], run_time=0.4)
        part_texts = VGroup(*[
            Text(f"m = {m:.2f}\nl = {l:.2f}", font=MONO, font_size=24, line_spacing=0.8).next_to(VGroup(*tokens[4 * c: 4 * c + 4]), DOWN, buff=0.4)
            for c, (m, l) in enumerate(parts)
        ])
        self.play(LaggedStart(*[FadeIn(p, shift=DOWN * 0.2) for p in part_texts], lag_ratio=0.3))

        m = max(p[0] for p in parts)
        l = sum(pl * math.exp(pm - m) for pm, pl in parts)
        merged = Text(f"merged:  m = {m:.2f}   l = {l:.2f}", font=MONO, font_size=30, color=YELLOW).move_to(DOWN * 2.2)
        arrows = VGroup(*[Arrow(p.get_bottom(), merged.get_top(), buff=0.1, stroke_width=3, color=GREY_B) for p in part_texts])
        rule = Text("m = max of the m's     l = Σ lᵢ · e^(mᵢ − m)", font=SERIF, font_size=24, color=GREY_B).next_to(merged, DOWN, buff=0.3)
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2), FadeIn(merged))
        self.play(FadeIn(rule))
        check = Text(f"one pass gave m = {full_m:.2f}, l = {full_l:.2f}. Same answer.", font=SERIF, font_size=26, color=TEAL_C)
        check.next_to(rule, DOWN, buff=0.25)
        self.play(FadeIn(check))
        self.wait(3)


class LatencyHiding(Scene):
    def construct(self):
        title = heading("Why one warp per head is slow")
        self.play(Write(title))
        x0, wait, math_w, row_h = -5.2, 1.9, 0.35, 0.32

        def row(y, offset, label):
            parts, x = [], x0 + offset
            while x + wait + math_w <= 6.4:
                parts.append(Rectangle(width=wait, height=row_h, stroke_width=0, fill_color=GREY_D, fill_opacity=1).move_to([x + wait / 2, y, 0]))
                parts.append(Rectangle(width=math_w, height=row_h, stroke_width=0, fill_color=YELLOW, fill_opacity=1).move_to([x + wait + math_w / 2, y, 0]))
                x += wait + math_w
            name = Text(label, font=MONO, font_size=18, color=GREY_B).next_to([x0, y, 0], LEFT, buff=0.25)
            return VGroup(*parts), name

        legend = VGroup(
            Rectangle(width=0.4, height=0.2, stroke_width=0, fill_color=GREY_D, fill_opacity=1),
            Text("waiting on a K/V load", font=SERIF, font_size=22),
            Rectangle(width=0.4, height=0.2, stroke_width=0, fill_color=YELLOW, fill_opacity=1),
            Text("math for one token", font=SERIF, font_size=22),
        ).arrange(RIGHT, buff=0.2).move_to(UP * 2.4)
        legend[2].shift(RIGHT * 0.4)
        legend[3].shift(RIGHT * 0.4)
        self.play(FadeIn(legend))

        bars, name = row(1.4, 0, "warp 0")
        self.play(FadeIn(name), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.6, run_time=3))
        cap = caption("Path A: every token waits on memory. 4,096 tokens take over a millisecond.")
        self.play(FadeIn(cap))
        self.wait(2)

        self.play(FadeOut(cap), Transform(title, heading("More warps hide the wait")))
        rows = [row(1.4 - 0.45 * k, 0.24 * k, f"warp {k}") for k in range(1, 7)]
        self.play(LaggedStart(*[AnimationGroup(FadeIn(n), *[GrowFromEdge(b, LEFT) for b in bs]) for bs, n in rows], lag_ratio=0.25, run_time=3))
        bus = Rectangle(width=11.6, height=0.32, stroke_width=0, fill_color=TEAL_C, fill_opacity=1).move_to([x0 + 5.8, -2.0, 0])
        bus_label = Text("memory bus", font=MONO, font_size=18, color=TEAL_C).next_to([x0, -2.0, 0], LEFT, buff=0.25)
        self.play(FadeIn(bus_label), GrowFromEdge(bus, LEFT), run_time=1.5)
        self.play(FadeIn(caption("While one warp waits, others load. Path B's chunks give the GPU these warps.")))
        self.wait(2.5)
