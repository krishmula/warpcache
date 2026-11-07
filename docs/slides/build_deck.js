// Builds docs/slides/warpcache-class-talk.pptx. Expected numbers come from
// scripts/expected_results.py, so rerun this after changing the model.
//
//   NODE_PATH=<dir with pptxgenjs> node docs/slides/build_deck.js

const path = require("path");
const { execFileSync } = require("child_process");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(__dirname, "warpcache-class-talk.pptx");
const APPLY_THEME = process.env.APPLY_THEME;

const model = JSON.parse(
  execFileSync("python3", [path.join(ROOT, "scripts/expected_results.py"), "--json"], { encoding: "utf8" })
);
const row = (batch, ctx) => model.rows.find((r) => r.batch === batch && r.ctx === ctx);
const us = (x) => Math.round(x.us);

const HEX = {
  ink: "171B22",
  slate: "2A3442",
  paper: "FFFFFF",
  mist: "EEF1F5",
  amber: "E3A23B",
  teal: "1F7A8C",
  rust: "C0533A",
  gray: "8792A2",
  sage: "6FB3A6",
  muted: "5B6675",
};

const THEME = {
  name: "warpcache",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.paper, dk2: HEX.slate, lt2: HEX.mist,
    accent1: HEX.teal, accent2: HEX.amber, accent3: HEX.rust,
    accent4: HEX.gray, accent5: HEX.sage, accent6: HEX.muted,
    hlink: HEX.teal, folHlink: HEX.muted,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Krishna Mula";
pres.title = "Workload-Aware CUDA Decode Attention over an INT8 Paged KV Cache";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "Dark",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.5, w: 8.8, h: 1.5, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.15, w: 8.8, h: 1.2, fontSize: 18, color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 7.6, h: 0.75, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "warpcache  ·  CS 259", options: { x: 0.5, y: 5.2, w: 4, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right", margin: 0 },
});

pres.defineSlideMaster({
  title: "Video",
  background: { color: "000000" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 7.6, h: 0.6, fontSize: 26, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "warpcache  ·  CS 259", options: { x: 0.5, y: 5.2, w: 4, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right", margin: 0 },
});

const MEDIA = path.join(__dirname, "media");

function videoSlide(sectionTitle, title, clip, notes) {
  const s = pres.addSlide({ masterName: "Video", sectionTitle });
  s.addText(title, { placeholder: "title" });
  const h = 4.2, w = h * 16 / 9;
  const cover = require("fs").readFileSync(path.join(MEDIA, `${clip}.png`)).toString("base64");
  s.addMedia({ type: "video", path: path.join(MEDIA, `${clip}.mp4`), cover: `image/png;base64,${cover}`, x: (10 - w) / 2, y: 0.95, w, h, objectName: `${clip} clip` });
  s.addNotes(notes);
  return s;
}

function expectedTag(slide) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 8.2, y: 0.45, w: 1.3, h: 0.42, rectRadius: 0.21, fill: { color: C.accent2 }, line: { color: C.accent2 },
    objectName: "Expected tag",
  });
  slide.addText("EXPECTED", {
    x: 8.2, y: 0.45, w: 1.3, h: 0.42, fontSize: 12, bold: true, color: C.text1, align: "center", valign: "middle",
    margin: 0, isTextBox: true, objectName: "Expected tag text",
  });
}

function measuredTag(slide) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 8.2, y: 0.45, w: 1.3, h: 0.42, rectRadius: 0.21, fill: { color: C.accent1 }, line: { color: C.accent1 },
    objectName: "Measured tag",
  });
  slide.addText("MEASURED", {
    x: 8.2, y: 0.45, w: 1.3, h: 0.42, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle",
    margin: 0, isTextBox: true, objectName: "Measured tag text",
  });
}

function card(slide, { x, y, w, h, fill, name }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { color: fill || C.background2 },
    objectName: name,
  });
}

function stat(slide, { x, y, w, big, label, color }) {
  slide.addText(big, { x, y, w, h: 0.9, fontSize: 44, bold: true, fontFace: THEME.headFontFace, color: color || C.text1, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.95, w, h: 0.8, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true });
}

const CHART_TEXT = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
const CHART_FRAME = {
  ...CHART_TEXT,
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  valGridLine: { color: "DDE2E8", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.slate,
  valAxisTitleColor: HEX.muted, valAxisTitleFontSize: 11, valAxisTitleFontFace: "+mn-lt",
};

// 1. Title
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "Dark", sectionTitle: "Opening" });
  s.addText("Workload-Aware CUDA Decode Attention over an INT8 Paged KV Cache", { placeholder: "title" });
  s.addText("Krishna Mula  ·  CS 259  ·  October 28, 2026", { placeholder: "body" });
  s.addNotes("One question: how do INT8 storage and the way we split work across the GPU interact for decode attention, and can a simple rule pick the right configuration per batch shape?");
}

// 2. Problem
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Opening" });
  s.addText("Decode attention is a memory problem", { placeholder: "title" });
  s.addText("One decode step, one layer of Qwen3-1.7B, 16K tokens of history, FP16 cache", {
    x: 0.5, y: 1.15, w: 9, h: 0.4, fontSize: 16, color: C.text2, margin: 0, isTextBox: true,
  });
  card(s, { x: 0.5, y: 1.8, w: 2.8, h: 2.3, name: "Bytes card" });
  card(s, { x: 3.6, y: 1.8, w: 2.8, h: 2.3, name: "Memory time card" });
  card(s, { x: 6.7, y: 1.8, w: 2.8, h: 2.3, name: "Math time card" });
  stat(s, { x: 0.75, y: 2.0, w: 2.3, big: "64 MiB", label: "of K and V read to produce one 128-number output per head" });
  stat(s, { x: 3.85, y: 2.0, w: 2.3, big: "72 µs", label: "to stream it at the RTX 3090's 936 GB/s peak", color: C.accent3 });
  stat(s, { x: 6.95, y: 2.0, w: 2.3, big: "3.8 µs", label: "for the 134 MFLOP of math at 35.6 FP32 TFLOPS", color: C.accent1 });
  s.addText("About 19 times more time moving bytes than doing math. Every design choice follows from that.", {
    x: 0.5, y: 4.4, w: 9, h: 0.5, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true,
  });
  s.addNotes("16 query heads x 16,384 tokens x 128 dims x 2 FLOP x 2 (QK and PV) = 134 MFLOP. Bytes = 16,384 tokens x 4 KiB per token per layer (8 KV heads x 128 dims x 2 bytes x K and V). These are arithmetic, not measurements.");
}

// 3. Two levers
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Opening" });
  s.addText("Two levers: fewer bytes, more of the chip", { placeholder: "title" });
  card(s, { x: 0.5, y: 1.35, w: 4.3, h: 3.5, name: "INT8 lever" });
  card(s, { x: 5.2, y: 1.35, w: 4.3, h: 3.5, name: "Split lever" });
  s.addText("INT8 cache", { x: 0.8, y: 1.55, w: 3.7, h: 0.45, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText("Split the history", { x: 5.5, y: 1.55, w: 3.7, h: 0.45, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText("Bytes per token per layer", { x: 0.8, y: 2.1, w: 3.7, h: 0.3, fontSize: 14, color: C.text2, margin: 0, isTextBox: true });
  const barX = 0.8, barW = 3.7;
  s.addShape(pres.shapes.RECTANGLE, { x: barX, y: 2.5, w: barW, h: 0.42, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: "FP16 bar" });
  s.addText("FP16  4,096 B", { x: barX + 0.1, y: 2.5, w: 3, h: 0.42, fontSize: 14, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true });
  s.addShape(pres.shapes.RECTANGLE, { x: barX, y: 3.05, w: barW * (2112 / 4096), h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "INT8 bar" });
  s.addText("INT8  2,112 B", { x: barX + 0.1, y: 3.05, w: 3, h: 0.42, fontSize: 14, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true });
  s.addText("Symmetric, one FP32 scale per token and KV head. 52% of the bytes, so up to 1.9x faster if dequantizing is cheap.", {
    x: 0.8, y: 3.65, w: 3.7, h: 1.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true,
  });
  s.addText("Batch 1 launches one block per query head", { x: 5.5, y: 2.1, w: 3.7, h: 0.3, fontSize: 14, color: C.text2, margin: 0, isTextBox: true });
  const cols = 14, cell = 0.22, gap = 0.04;
  for (let i = 0; i < 82; i++) {
    const cx = 5.5 + (i % cols) * (cell + gap);
    const cy = 2.5 + Math.floor(i / cols) * (cell + gap);
    const busy = i < 16;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: cell, h: cell, fill: { color: busy ? C.accent3 : "FFFFFF" }, line: { color: busy ? C.accent3 : C.accent4, width: 0.75 }, objectName: `SM ${i}` });
  }
  s.addText("16 of 82 SMs busy. Splitting each head's history into chunks gives every SM work, at the cost of a merge.", {
    x: 5.5, y: 4.15, w: 3.7, h: 0.65, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true,
  });
}

pres.addSection({ title: "GPU primer" });
videoSlide("GPU primer", "Threads, lanes, and warps", "ThreadsAndWarps",
  "A thread is one stream of instructions. On NVIDIA GPUs threads run in groups of 32 called warps, and the 32 lanes of a warp execute the same instruction at the same time. In our kernel, one warp handles one query head, and each lane owns 4 of the 128 numbers in q, K, and V.");
videoSlide("GPU primer", "A warp adds 32 numbers in 5 steps", "WarpShuffle",
  "__shfl_xor_sync lets a lane read another lane's register directly, without shared memory. In round r, lane i reads from lane i xor d, for d = 16, 8, 4, 2, 1. After 5 rounds every lane has the full sum. This is how each token's score q · k is finished, and it is the warp_reduce warm-up kernel.");
videoSlide("GPU primer", "Blocks, the grid, and 82 SMs", "GridBlocksSMs",
  "A block is a group of threads that runs on one streaming multiprocessor (SM). Our blocks are a single warp. The grid is every block in a launch: one per request and query head. At batch 1 that is 16 blocks on a chip with 82 SMs. Splitting the history into chunks multiplies the number of blocks. The real chunk counts are larger, and each SM runs up to 16 of our blocks at once.");

// 4. Design
pres.addSection({ title: "Design" });
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Design" });
  s.addText("One kernel, two paths", { placeholder: "title" });
  s.addText("decode(q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, num_chunks)", {
    x: 0.5, y: 1.1, w: 9, h: 0.35, fontSize: 13, fontFace: "Courier New", color: C.text2, margin: 0, isTextBox: true,
  });
  const lane = (y, label, chunks, color, tail) => {
    s.addText(label, { x: 0.5, y, w: 1.9, h: 0.9, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
    const x0 = 2.5, total = 4.6, g = 0.08;
    const w = (total - g * (chunks - 1)) / chunks;
    for (let i = 0; i < chunks; i++) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0 + i * (w + g), y: y + 0.15, w, h: 0.6, rectRadius: 0.06, fill: { color }, line: { color }, objectName: `${label} chunk ${i}` });
    }
    s.addShape(pres.shapes.RIGHT_ARROW, { x: 7.2, y: y + 0.3, w: 0.45, h: 0.3, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: `${label} arrow` });
    s.addText(tail, { x: 7.75, y, w: 1.75, h: 0.9, fontSize: 13, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  };
  lane(1.65, "Path A\n1 chunk", 1, C.accent3, "normalize, write output");
  lane(2.75, "Path B\nN chunks", 6, C.accent1, "(m, l, u) per chunk, merged in PyTorch");
  s.addText("Grid: (request, query head, chunk). Templated on the cache type, so INT8 and FP16 share one source.", {
    x: 0.5, y: 3.95, w: 9, h: 0.35, fontSize: 14, color: C.text1, margin: 0, isTextBox: true,
  });
  s.addText("Merge two partial states:  m = max(m₁, m₂),  l = l₁·e^(m₁−m) + l₂·e^(m₂−m),  u likewise,  output = u / l", {
    x: 0.5, y: 4.4, w: 9, h: 0.4, fontSize: 14, color: C.text2, margin: 0, isTextBox: true,
  });
  s.addNotes("Each block runs online softmax over its token range: a running max m, a running sum l, and a running weighted sum of V rows u. Path B blocks stop before normalizing and write (m, l, u). An empty chunk writes m = -inf, l = 0, u = 0, and the merge handles it exactly.");
}

videoSlide("Design", "Why chunks can be merged exactly", "OnlineSoftmax",
  "Softmax needs the max and the sum over all scores, but online softmax keeps them as a running state: m, the largest score so far, and l, the sum of e^(z - m). V gets the same running treatment in u. Any split of the history into chunks gives partial states that merge to the exact one-pass answer. That is what makes Path B correct.");

// 5. Setup
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Design" });
  s.addText("What the numbers will be measured on", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  s.addTable([
    [hdr("Item"), hdr("Choice")],
    ["GPU", "RTX 3090: 82 SMs, 24 GB, 936 GB/s, 6 MB L2 (Vast.ai VM, Nsight counters work)"],
    ["Model shape", "Qwen3-1.7B: 16 query heads, 8 KV heads, head dim 128"],
    ["Cache", "Paged, 16 tokens per page. INT8 with per-token scales, or FP16"],
    ["Workloads", "Batch 1, 4, 16 at contexts 1K, 4K, 16K. Decode only"],
    ["Baseline", "PyTorch SDPA on a pre-gathered contiguous FP16 history"],
    ["Timing", "CUDA events, warmup, repeats, working set larger than L2"],
  ], {
    x: 0.5, y: 1.3, w: 9, colW: [1.8, 7.2], fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.5, color: "D5DBE3" },
    fill: { color: C.background1 }, rowH: 0.42, valign: "middle", margin: [0, 0.1, 0, 0.1],
  });
  s.addNotes("SDPA is timed on a history that is already contiguous, which excludes the gather. That makes it the harder baseline to beat. Out of scope: prefill, mixed-length batches, FlashInfer, a CUDA merge kernel, and vLLM integration.");
}

// 6. Status
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Design" });
  s.addText("Where the project stands", { placeholder: "title" });
  const col = (x, title, color, items) => {
    card(s, { x, y: 1.3, w: 2.85, h: 3.6, name: `${title} column` });
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.5, w: 0.3, h: 0.3, fill: { color }, line: { color }, objectName: `${title} dot` });
    s.addText(title, { x: x + 0.65, y: 1.45, w: 2.0, h: 0.4, fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
    s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })), {
      x: x + 0.25, y: 2.0, w: 2.4, h: 2.8, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true,
    });
  };
  col(0.5, "Done", C.accent1, [
    "PyTorch reference and 26 tests",
    "Online softmax and merge in Python",
    "Warp sum and max kernel",
    "GPU check: Nsight works on the 3090",
  ]);
  col(3.575, "In progress", C.accent2, [
    "Path A FP16 kernel, one warp per head",
    "Kernel tests against the reference",
  ]);
  col(6.65, "Next", C.accent4, [
    "INT8 through the template",
    "Path B with a PyTorch merge",
    "Benchmarks, Nsight, selector rule",
    "Paper by November 5",
  ]);
}

// 7. Expected divider
pres.addSection({ title: "Expected results" });
{
  const s = pres.addSlide({ masterName: "Dark", sectionTitle: "Expected results" });
  s.addText("Expected results", { placeholder: "title" });
  s.addText("Predictions from a performance model of the kernel as designed. Not measured yet. The goal is to match or explain each one.", { placeholder: "body" });
  s.addNotes("The model is scripts/expected_results.py. Each path's time is the larger of two limits: streaming the bytes at 85% of peak bandwidth (the GPU check reached 87%), or the time one warp needs to walk its tokens one at a time. The per-token cost, 0.3 microseconds, is the guess that matters most. Its plausible range is 0.15 to 0.6.");
}

videoSlide("Expected results", "Why Path A should be slow", "LatencyHiding",
  "A global memory load takes hundreds of nanoseconds. One warp walking tokens in order spends most of its time waiting, so the memory bus sits mostly idle. With many warps resident, the SM switches to another warp while one waits, which keeps the bus busy. Path B provides those warps. This is the reasoning behind the expected numbers that follow.");

// 8. Batch 1 context sweep
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Expected results" });
  s.addText("Batch 1: splitting is worth 25 to 44x", { placeholder: "title" });
  expectedTag(s);
  const ctxs = [1024, 4096, 16384];
  const labels = ["1K", "4K", "16K"];
  const series = [
    { name: "Path A, FP16", values: ctxs.map((c) => us(row(1, c).a_fp16)) },
    { name: "Path B, FP16", values: ctxs.map((c) => us(row(1, c).b_fp16)) },
    { name: "Path B, INT8", values: ctxs.map((c) => us(row(1, c).b_int8)) },
    { name: "SDPA, FP16", values: ctxs.map((c) => us(row(1, c).sdpa_fp16)) },
  ].map((x) => ({ ...x, labels }));
  s.addChart(pres.charts.LINE, series, {
    x: 0.4, y: 1.15, w: 5.9, h: 3.9, ...CHART_FRAME,
    chartColors: [HEX.rust, HEX.teal, HEX.sage, HEX.gray], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    valAxisLogScaleBase: 10, valAxisMinVal: 10, valAxisMaxVal: 10000, valAxisTitle: "latency, µs (log scale)", showValAxisTitle: true,
    showCatAxisTitle: true, catAxisTitle: "context length", catAxisTitleColor: HEX.muted, catAxisTitleFontSize: 11, catAxisTitleFontFace: "+mn-lt",
  });
  const a4 = us(row(1, 4096).a_fp16), b4 = us(row(1, 4096).b_fp16), sd4 = us(row(1, 4096).sdpa_fp16);
  s.addText([
    { text: `Path A ${(a4 / 1000).toFixed(1)} ms at 4K`, options: { bold: true, breakLine: true } },
    { text: "One warp walks all 4,096 tokens in order. Its time is per-token latency, about 1% of DRAM bandwidth.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: `Path B ${b4} µs, SDPA ${sd4} µs`, options: { bold: true, breakLine: true } },
    { text: "Chunks fill the chip. At batch 1 the PyTorch merge (about 25 µs) keeps Path B behind SDPA. FP16 and INT8 overlap here." },
  ], { x: 6.5, y: 1.3, w: 3.0, h: 3.6, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Path A grows linearly with context and does not depend on bandwidth, because one warp can have only a few loads in flight. If the per-token cost is 0.15 us instead of 0.3, Path A halves. The ranking does not change.");
}

// 9. Batch 16
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Expected results" });
  const r = row(16, 16384);
  const gain = (r.sdpa_fp16.us / r.b_int8.us).toFixed(1);
  s.addText(`At batch 16, 16K: INT8 beats SDPA ${gain}x`, { placeholder: "title" });
  expectedTag(s);
  const cats = ["B4 · 4K", "B4 · 16K", "B16 · 4K", "B16 · 16K"];
  const pts = [[4, 4096], [4, 16384], [16, 4096], [16, 16384]];
  const series = [
    { name: "SDPA, FP16", labels: cats, values: pts.map(([b, c]) => us(row(b, c).sdpa_fp16)) },
    { name: "Path B, FP16", labels: cats, values: pts.map(([b, c]) => us(row(b, c).b_fp16)) },
    { name: "Path B, INT8", labels: cats, values: pts.map(([b, c]) => us(row(b, c).b_int8)) },
  ];
  s.addChart(pres.charts.BAR, series, {
    x: 0.4, y: 1.2, w: 6.2, h: 3.85, barDir: "col", barGapWidthPct: 60, ...CHART_FRAME,
    chartColors: [HEX.gray, HEX.teal, HEX.sage], valAxisTitle: "latency, µs", showValAxisTitle: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: HEX.slate,
  });
  s.addText([
    { text: "Where bytes dominate, INT8 pays off", options: { bold: true, breakLine: true } },
    { text: `At batch 16 and 16K: SDPA ${us(r.sdpa_fp16)} µs, Path B FP16 ${us(r.b_fp16)} µs, Path B INT8 ${us(r.b_int8)} µs.`, options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Path A is not shown. It stays near 1.2 ms at 4K and 4.9 ms at 16K for every batch size." },
  ], { x: 6.8, y: 1.3, w: 2.7, h: 3.6, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}

// 10. Why INT8 is not 1.9x
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Expected results" });
  s.addText("INT8 should win 1.0 to 1.3x, not 1.9x", { placeholder: "title" });
  expectedTag(s);
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  const line = (b, c) => {
    const r = row(b, c);
    return [
      `B${b} · ${c / 1024}K`,
      `${us(r.b_fp16)} µs`,
      `${us(r.b_int8)} µs`,
      `${(r.b_fp16.us / r.b_int8.us).toFixed(2)}x`,
      `${Math.round(r.b_fp16.dram_pct)}% / ${Math.round(r.b_int8.dram_pct)}%`,
    ];
  };
  s.addTable([
    [hdr("Shape"), hdr("Path B FP16"), hdr("Path B INT8"), hdr("INT8 gain"), hdr("DRAM % of peak, FP16 / INT8")],
    line(1, 16384), line(4, 4096), line(4, 16384), line(16, 4096), line(16, 16384),
  ], {
    x: 0.5, y: 1.25, w: 9, colW: [1.4, 1.6, 1.6, 1.4, 3.0], fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.5, color: "D5DBE3" },
    fill: { color: C.background1 }, rowH: 0.4, valign: "middle", align: "center", margin: [0, 0.08, 0, 0.08],
  });
  card(s, { x: 0.5, y: 3.85, w: 9, h: 1.05, name: "Limiter callout" });
  s.addText("Halving the bytes moves the limit from DRAM to latency. With one warp per head, each token still costs a load, five shuffles, and two exponentials, so INT8 Path B hits about 50% of peak bandwidth while FP16 reaches about 80%.", {
    x: 0.75, y: 3.95, w: 8.5, h: 0.85, fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true,
  });
  s.addNotes("This is the prediction most worth testing with Nsight Compute. If INT8 Path B shows about 50% DRAM throughput and high stall-long-scoreboard, the latency explanation holds. The fix for later is more warps per head, so several tokens are in flight at once.");
}

// 11. Selector
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Expected results" });
  s.addText("The selector becomes a chunk-count rule", { placeholder: "title" });
  expectedTag(s);
  const steps = [
    ["Batch 1", `${row(1, 4096).b_fp16.chunks} chunks`, "1,024 warps"],
    ["Batch 4", `${row(4, 4096).b_fp16.chunks} chunks`, "1,024 warps"],
    ["Batch 16", `${row(16, 4096).b_fp16.chunks} chunks`, "1,024 warps"],
  ];
  steps.forEach(([b, c, w], i) => {
    const x = 0.5 + i * 3.1;
    card(s, { x, y: 1.35, w: 2.8, h: 2.0, name: `${b} card` });
    s.addText(b, { x: x + 0.25, y: 1.5, w: 2.3, h: 0.4, fontSize: 16, color: C.text2, margin: 0, isTextBox: true });
    s.addText(c, { x: x + 0.25, y: 1.95, w: 2.3, h: 0.8, fontSize: 28, bold: true, fontFace: THEME.headFontFace, color: C.accent1, margin: 0, isTextBox: true });
    s.addText(w, { x: x + 0.25, y: 2.8, w: 2.3, h: 0.4, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
  });
  s.addText([
    { text: "With a one-warp Path A, splitting is expected to win at every shape in the sweep. ", options: {} },
    { text: "The rule picks chunks so that batch × 16 heads × chunks ≈ 1,024 warps, close to the 1,312 the 3090 can keep resident.", options: { bold: true } },
  ], { x: 0.5, y: 3.65, w: 9, h: 0.9, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("The original question was A versus B. The model says Path A as written never wins, so the interesting decision is the chunk count. If Path A gets more warps per head later, the A versus B crossover comes back at large batch.");
}

// 12. Measured quantization error
pres.addSection({ title: "Measured so far" });
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Measured so far" });
  s.addText("Per-token INT8 is lossy because of K", { placeholder: "title" });
  measuredTag(s);
  s.addChart(pres.charts.BAR, [
    { name: "Mean", labels: ["Qwen2.5-1.5B", "Qwen3-1.7B"], values: [2.7, 6.1] },
    { name: "Worst", labels: ["Qwen2.5-1.5B", "Qwen3-1.7B"], values: [40, 33] },
  ], {
    x: 0.4, y: 1.2, w: 5.4, h: 3.85, barDir: "col", barGapWidthPct: 60, ...CHART_FRAME,
    chartColors: [HEX.teal, HEX.rust], valAxisTitle: "relative L2 error of attention output, %", showValAxisTitle: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.slate, dataLabelFormatCode: "0.0",
  });
  s.addText([
    { text: "V quantizes well, about 1% error.", options: { bullet: true, breakLine: true } },
    { text: "K has outlier channels, up to 66x the median, so one scale per token wastes precision on the rest.", options: { bullet: true, breakLine: true } },
    { text: "Static K smoothing (QServe) cuts Qwen3 to 1.8% mean, 8% worst. It is future work.", options: { bullet: true } },
  ], { x: 6.0, y: 1.3, w: 3.5, h: 3.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
  s.addText("CPU, one 2,048-token text, 16 decode positions. A small sample.", {
    x: 6.0, y: 4.45, w: 3.5, h: 0.5, fontSize: 11, color: C.accent6, margin: 0, valign: "top", isTextBox: true,
  });
}

// 13. Plan
pres.addSection({ title: "Closing" });
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Closing" });
  s.addText("From expected to measured", { placeholder: "title" });
  const steps = [
    ["1", "Path A correct", "FP16, then INT8 through the template. Zero compute-sanitizer errors."],
    ["2", "Path B correct", "Chunks plus PyTorch merge match the reference, including empty chunks."],
    ["3", "Measure", "Latency at 9 shapes, Nsight on 2 to 4 cases. Check the 0.3 µs per-token guess first."],
    ["4", "Explain", "Every gap between expected and measured gets a cause, backed by a profile."],
  ];
  steps.forEach(([n, head, body], i) => {
    const y = 1.35 + i * 0.95;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `Step ${n} circle` });
    s.addText(n, { x: 0.5, y, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(head, { x: 1.3, y, w: 2.4, h: 0.6, fontSize: 18, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(body, { x: 3.8, y, w: 5.7, h: 0.6, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  });
  s.addNotes("Cut from the class scope, and the paper says so: a CUDA cache-write kernel, FlashInfer, K smoothing, GQA head sharing, mixed-length batches, a CUDA merge kernel, and vLLM integration.");
}

// 14. Questions
{
  const s = pres.addSlide({ masterName: "Dark", sectionTitle: "Closing" });
  s.addText("Questions", { placeholder: "title" });
  s.addText("github.com/krishmula/warpcache", { placeholder: "body" });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log(OUT);
})();
