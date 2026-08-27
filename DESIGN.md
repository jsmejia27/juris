# Design System: Juris Philippine Legal AI Platform

## 1. Visual Theme & Atmosphere
An authoritative, editorial judicial atmosphere married to a responsive, high-precision legal research engine. The visual personality balances the gravitas of the Supreme Court chambers with the crisp clarity of a modern institutional workspace. Generous spatial separation, restrained hairline dividers, and deliberate typographic weight replace decorative clutter. 

- **Density:** 6/10 (Balanced research layout with monospaced citations)
- **Variance:** 7/10 (Asymmetric split-screen hero with integrated live streaming assistant)
- **Motion:** 5/10 (Spring physics on interactive elements, subtle token stream reveals, zero jitter)

---

## 2. Color Palette & Roles
- **Midnight Judicial Base** (`#0a0f1d`) — Deep navy header, primary buttons, and terminal anchors.
- **Canvas Off-White** (`#f8fafc`) — Primary document canvas and page background.
- **Card Surface** (`#ffffff`) — Elevated white cards and workspace containers.
- **Charcoal Ink** (`#0f172a`) — Primary body headlines and high-contrast titles.
- **Slate Text** (`#475569`) — Secondary explanatory copy and legal descriptions.
- **Whisper Hairline** (`#e2e8f0`) — 1px clean container borders and dividers.
- **Ochre Gold Accent** (`#b8860b`) — Single accent for judicial seal, active focus rings, and citation badges. (Saturation: 74%).
- **Banned:** Neon purple/cyan gradients, pure black (`#000000`), fluorescent outer glow shadows.

---

## 3. Typography Rules
- **Display / Headers:** `Playfair Display` — Track-tight, controlled scale, editorial judicial hierarchy.
- **Body Copy:** `Plus Jakarta Sans` — Relaxed leading (1.6), maximum 65 characters per line for legal readability.
- **Monospace / Citations:** `JetBrains Mono` — For Republic Act numbers, G.R. docket citations, section references, and search latency metrics.
- **Banned:** `Inter` (generic default), generic system serifs (`Times New Roman`, `Georgia`).

---

## 4. Component Behaviors
- **Integrated Assistant:** Embedded directly into the landing page Hero section as an asymmetric split-screen workspace with live token streaming, thought disclosure, and interactive citation popovers.
- **Buttons:** Tactile -1px transform on `:active`. Solid `#0a0f1d` with `#f8fafc` text for primary; 1px `#e2e8f0` border for secondary.
- **Citations & Badges:** Monospaced pill chips with 1px `#e2e8f0` border, subtly highlighted with ochre gold on hover.
- **Inputs:** Minimalist hairline input with `#b8860b` focus ring.

---

## 5. Layout Architecture
- **Hero Split-Screen:** 
  - *Left Column (45%):* Executive value proposition, dataset fidelity counters, and 1-click legal prompt chips.
  - *Right Column (55%):* The **Live Juris Legal Assistant Interface** embedded in-place with real-time SSE streaming, category filter selector, verbatim source cards, and official PDF downloads.
- **Corpus Pillars (2-Column Zig-Zag):** Detailed breakdowns of structured Philippine legal analysis, 20th Congress live sync, and Standalone Native Qdrant vector retrieval.
- **Mobile Responsive (<768px):** Clean single-column collapse with 100% full-width chat container and 44px touch targets.

---

## 6. Anti-Patterns (Banned)
- ❌ No emojis in headers or formal UI chrome.
- ❌ No generic "3 equal cards" layout.
- ❌ No invented/fake statistics. All metrics reflect actual indexed corpus (12,000+ Republic Acts, 68,000+ Jurisprudence cases, <30ms retrieval).
- ❌ No "AI purple/blue" neon glows or floating particles.
- ❌ No decorative filler text ("Scroll to explore").
