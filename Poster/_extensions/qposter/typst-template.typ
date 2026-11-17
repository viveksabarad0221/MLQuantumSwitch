// ============================================================================
//  QPoster — A0 portrait Typst poster template for Quarto
//  Two themes, chosen with `poster-theme:` in the YAML header
//    paper : all-serif, physics-paper look, hairline column rules
//    swiss : Helvetica-style headings over a serif text face, heavy rules
// ============================================================================

#let themes = (
  paper: (
    head-font: "Libertinus Serif",
    body-font: "Libertinus Serif",
    math-font: "Libertinus Math",
    size: 30pt,
    ink: rgb("#111111"),
    muted: rgb("#555555"),
    accent: rgb("#8C1D18"),   // oxblood — also causal order B→A
    second: rgb("#1F4A8A"),   // ink blue — causal order A→B
    rule: rgb("#111111"),
    tint: rgb("#F4F1EC"),
    justify: true,
  ),
  swiss: (
    head-font: "TeX Gyre Heros",
    body-font: "Libertinus Serif",
    math-font: "Libertinus Math",
    size: 30pt,
    ink: rgb("#111111"),
    muted: rgb("#555555"),
    accent: rgb("#D22B26"),   // signal red — causal order B→A
    second: rgb("#1F4A8A"),   // ink blue — causal order A→B
    rule: rgb("#111111"),
    tint: rgb("#EDEDED"),
    justify: false,
  ),
)

#let theme-state = state("qposter-theme", themes.paper)

// Inline colouring by causal order:  [text]{.ab}  /  [text]{.ba}
#let ab(body) = context text(fill: theme-state.get().second, body)
#let ba(body) = context text(fill: theme-state.get().accent, body)

// --------------------------------------------------------------------------
//  Section ("card" in the qmd)
// --------------------------------------------------------------------------
#let section-counter = counter("qposter-section")

#let pcard(title: none, accent: none, numbered: true, body) = context {
  let t = theme-state.get()
  if numbered { section-counter.step() }
  let num = if numbered { context section-counter.display() } else { none }
  let has-title = title != none and title != []

  block(width: 100%, breakable: false, above: 0pt, below: 2.2cm)[
    #if has-title {
      if t.head-font == t.body-font {
        // paper: small-caps heading with a hairline underneath
        block(below: 0.8cm, {
          text(size: 50pt, fill: t.ink, {
            if num != none { text(fill: t.accent, num); h(0.5em) }
            smallcaps(title)
          })
          v(-0.15cm)
          line(length: 100%, stroke: 1.5pt + t.rule)
        })
      } else {
        // swiss: heavy rule above, bold grotesque heading
        block(below: 1cm, {
          line(length: 100%, stroke: 7pt + t.rule)
          v(0.35cm)
          text(font: t.head-font, size: 46pt, weight: "bold", fill: t.ink, tracking: -0.5pt, {
            if num != none { text(fill: t.accent, num); h(0.45em) }
            title
          })
        })
      }
    }
    #body
  ]
}

#let keybox(body) = context {
  let t = theme-state.get()
  set par(justify: false)
  if t.head-font == t.body-font {
    block(width: 100%, stroke: 2.5pt + t.accent, inset: (x: 1.1cm, y: 0.9cm),
      above: 1cm, below: 1cm, breakable: false, body)
  } else {
    block(width: 100%, fill: t.tint, inset: (x: 1.1cm, y: 1cm),
      above: 1cm, below: 1cm, breakable: false, {
        show strong: set text(fill: t.accent)
        body
      })
  }
}

#let note(accent: none, body) = context {
  let t = theme-state.get()
  set par(justify: false)
  block(width: 100%, inset: (left: 1cm, y: 0.3cm), stroke: (left: 3pt + t.muted),
    above: 0.9cm, below: 0.9cm, breakable: false, text(size: 0.92em, body))
}

#let placeholder(height: 15cm, body) = context {
  let t = theme-state.get()
  block(width: 100%, height: height, above: 0.8cm, below: 0.8cm,
    stroke: (paint: t.muted.lighten(40%), thickness: 2pt, dash: "dashed"),
    align(center + horizon, pad(2cm, text(fill: t.muted, size: 24pt, style: "italic", body))))
}

#let poster-grid(widths: (1fr, 1fr, 1fr), ..cols) = context {
  let t = theme-state.get()
  if t.head-font == t.body-font {
    // hairline rules between columns
    grid(columns: widths, column-gutter: 0pt,
      inset: (x, y) => (left: if x == 0 { 0pt } else { 1.3cm }, right: if x == widths.len() - 1 { 0pt } else { 1.3cm }),
      stroke: (x, y) => if x > 0 { (left: 1pt + t.muted.lighten(30%)) },
      ..cols.pos())
  } else {
    block(width: 100%, above: 0pt, below: 1.4cm, grid(columns: widths, column-gutter: 2.2cm, ..cols.pos()))
  }
}

#let twocol(widths: (1fr, 1fr), ..cols) = grid(columns: widths, column-gutter: 1cm, ..cols.pos())

// --------------------------------------------------------------------------
//  Narrative structure: lead sentence, layers on a spine, bridges
// --------------------------------------------------------------------------
#let spine-w = 9.5cm        // width of the left rail holding layer numbers
#let spine-gap = 1.8cm      // distance from the spine line to the content
#let spine-stroke = 4pt

#let lead(body) = context {
  let t = theme-state.get()
  block(width: 100%, above: 0pt, below: 1.4cm, inset: (left: spine-w + spine-gap),
    text(font: t.head-font, size: 44pt, fill: t.ink, tracking: -0.5pt, {
      set par(justify: false, leading: 0.4em)
      show strong: set text(fill: t.accent)
      body
    }))
}

#let layer(num: none, label: none, title: none, body) = context {
  let t = theme-state.get()
  block(above: 0pt, below: 0pt, grid(
    columns: (spine-w, 1fr),
    grid.cell(inset: (right: 1.2cm), align(right + top, {
      text(font: t.head-font, size: 110pt, weight: "bold", fill: t.accent, tracking: -4pt, top-edge: "cap-height", num)
      v(0.5cm)
      text(font: t.head-font, size: 24pt, weight: "bold", tracking: 3pt, fill: t.ink, upper(label))
    })),
    grid.cell(stroke: (left: spine-stroke + t.ink), inset: (left: spine-gap, bottom: 0.4cm), {
      if title != none and title != [] {
        block(below: 0.8cm, text(font: t.head-font, size: 54pt, weight: "bold", fill: t.ink,
          tracking: -1pt, top-edge: "cap-height", {
            set par(justify: false, leading: 0.3em)
            title
          }))
      }
      body
    }),
  ))
}

#let bridge(body) = context {
  let t = theme-state.get()
  let tw = 1.6cm
  block(above: 0pt, below: 0pt, grid(
    columns: (spine-w, 1fr),
    [],
    grid.cell(stroke: (left: spine-stroke + t.ink), inset: (left: spine-gap, y: 0.8cm), {
      place(left + horizon, dx: -spine-gap - tw / 2,
        polygon(fill: t.accent, (0cm, 0cm), (tw, 0cm), (tw / 2, 1.3cm)))
      text(size: 40pt, style: "italic", fill: t.ink, {
        set par(justify: false)
        body
      })
    }),
  ))
}

#let panel(title: none, body) = context {
  let t = theme-state.get()
  block(width: 100%, breakable: false, above: 0pt, below: 1cm, {
    if title != none and title != [] {
      line(length: 100%, stroke: 2pt + t.ink)
      v(0.35cm)
      block(below: 0.45cm, text(font: t.head-font, size: 34pt, weight: "bold", fill: t.ink, title))
    }
    body
  })
}

// --------------------------------------------------------------------------
//  Two tracks: abstract concept  <->  physical realisation, row by row
// --------------------------------------------------------------------------
#let connector(label) = context {
  let t = theme-state.get()
  stack(dir: ttb, spacing: 0.25cm,
    if label != "" { text(font: t.head-font, size: 40pt, weight: "bold", fill: t.accent, label) },
    text(size: 64pt, fill: t.ink, sym.arrow.l.r.long),
  )
}

#let tracks(left-title: [Theory], right-title: [Photonics], ..pairs) = context {
  let t = theme-state.get()
  let head(body) = text(font: t.head-font, size: 28pt, weight: "bold", tracking: 3pt, fill: t.muted, upper(body))
  let rows = pairs.pos().map(((l, lab, r)) => (
    grid.cell(inset: (y: 0.5cm, right: 0.6cm), align: left + horizon, l),
    grid.cell(inset: (y: 0.3cm), align: center + horizon, connector(lab)),
    grid.cell(inset: (y: 0.5cm, left: 0.6cm), align: left + horizon, r),
    grid.hline(stroke: 0.75pt + t.muted.lighten(50%)),
  )).flatten()
  block(width: 100%, above: 1.1cm, below: 0.4cm, grid(
    columns: (1fr, 6cm, 1fr),
    grid.cell(inset: (bottom: 0.5cm), head(left-title)),
    [],
    grid.cell(inset: (bottom: 0.5cm, left: 0.6cm), head(right-title)),
    grid.hline(stroke: 1.5pt + t.ink),
    ..rows,
  ))
}

// shaded flow box: ::: {.col .hl}
#let hl(body) = grid.cell(fill: rgb("#EDEDED"), body)

#let flow(widths: none, ..cells) = context {
  let t = theme-state.get()
  let items = cells.pos()
  let n = items.len()
  let cols = if widths != none { widths } else {
    range(2 * n - 1).map(i => if calc.even(i) { 1fr } else { auto })
  }
  let arrow = text(font: t.head-font, size: 46pt, weight: "bold", fill: t.accent, sym.arrow.r)
  let children = ()
  for (i, c) in items.enumerate() {
    children.push(c)
    if i < n - 1 { children.push(arrow) }
  }
  block(width: 100%, above: 0.7cm, below: 0.9cm, breakable: false, {
    set text(size: 0.85em)
    set par(justify: false, spacing: 0.35em)
    grid(
      columns: cols, align: center + horizon,
      inset: (x, y) => if calc.even(x) { (x: 0.6cm, y: 0.6cm) } else { (x: 0.35cm) },
      stroke: (x, y) => if calc.even(x) { 1.5pt + t.ink },
      ..children,
    )
  })
}

// --------------------------------------------------------------------------
//  Header and footer
// --------------------------------------------------------------------------
#let author-line(t, authors) = authors.map(a =>
  box[#a.name#super(text(fill: t.accent, a.aff))]).join([, #h(0.1em)])

#let affil-lines(affiliations) = affiliations.enumerate(start: 1).map(((i, a)) => [#super(str(i))#a]).join(linebreak())

#let poster-header(t, title, subtitle, event, authors, affiliations, logo) = {
  set par(justify: false)
  set text(hyphenate: false)
  if t.head-font == t.body-font {
    // paper: centred title block, logo left, event right, double rule
    pad(x: 3cm, top: 2.8cm, bottom: 0cm, {
      grid(
        columns: (19cm, 1fr, 19cm), column-gutter: 1cm, align: (left + top, center + top, right + top),
        if logo != none { image(logo, width: 20cm) },
        {
          text(size: 96pt, weight: "semibold", fill: t.ink)[
            #set par(leading: 0.3em, justify: false)
            #title
          ]
          if subtitle != none { v(0.3cm); text(size: 44pt, style: "italic", subtitle) }
          v(1.2cm)
          text(size: 40pt, fill: t.ink, author-line(t, authors))
          v(0.6cm)
          text(size: 22pt, style: "italic", fill: t.muted)[
            #set par(leading: 0.45em, justify: false)
            #affil-lines(affiliations)
          ]
        },
        if event != none { text(size: 30pt, fill: t.muted, smallcaps(event)) },
      )
      v(1.4cm)
      line(length: 100%, stroke: 3pt + t.rule)
      v(-0.55cm)
      line(length: 100%, stroke: 1pt + t.rule)
    })
  } else {
    // swiss: event + logo on top row, full-width grotesque title below, heavy rule
    block(width: 100%, height: 1.2cm, fill: t.ink, below: 0pt)
    pad(x: 3cm, top: 2cm, bottom: 0cm, {
      grid(
        columns: (1fr, auto), align: (left + horizon, right + horizon),
        if event != none {
          text(font: t.head-font, size: 28pt, weight: "bold", fill: t.accent, upper(event))
        },
        if logo != none { image(logo, width: 17cm) },
      )
      v(1.2cm)
      block(width: 100%, text(font: t.head-font, size: 102pt, weight: "bold", fill: t.ink, tracking: -2pt)[
        #set par(leading: 0.2em, justify: false)
        #title
      ])
      if subtitle != none { v(0.3cm); text(font: t.head-font, size: 48pt, subtitle) }
      v(1.4cm)
      text(size: 42pt, fill: t.ink, author-line(t, authors))
      v(0.6cm)
      text(size: 23pt, fill: t.muted)[
        #set par(leading: 0.45em, justify: false)
        #affil-lines(affiliations)
      ]
      v(1.4cm)
      line(length: 100%, stroke: 1.5pt + t.rule)
    })
  }
}

#let poster-footer(t, logos, funding, contact) = pad(x: 3cm, bottom: 1.6cm, {
  line(length: 100%, stroke: 1pt + t.rule)
  v(0.9cm)
  grid(
    columns: (auto, 1fr, auto), column-gutter: 2.5cm, align: horizon,
    stack(dir: ltr, spacing: 2.2cm, ..logos.map(l => image(l, height: 3.4cm))),
    text(size: 21pt, fill: t.muted, funding),
    if contact != none { text(size: 23pt, fill: t.ink, contact) },
  )
})

// --------------------------------------------------------------------------
//  Main template
// --------------------------------------------------------------------------
#let poster(
  title: none, subtitle: none, event: none,
  authors: (), affiliations: (), logo: none,
  footer-logos: (), funding: none, contact: none,
  theme: "paper", fontsize: none,
  doc,
) = {
  let t = themes.at(theme)
  if fontsize != none { t.size = fontsize }
  theme-state.update(t)

  set page(width: 841mm, height: 1189mm, margin: 0pt, fill: white)
  set text(font: t.body-font, size: t.size, fill: t.ink, lang: "en",
    hyphenate: t.justify, number-type: "lining")
  set par(justify: t.justify, leading: 0.55em, spacing: 0.9em)
  show math.equation: set text(font: t.math-font)
  show math.equation.where(block: true): set block(above: 0.8em, below: 0.8em)
  set list(indent: 0.3em, body-indent: 0.5em, marker: [–])
  set enum(indent: 0.3em, body-indent: 0.5em)
  show link: set text(fill: t.second)

  // tables: booktabs-style rules, grotesque header
  set table(
    stroke: (x, y) => (
      top: if y == 0 { 2.5pt + t.ink } else if y == 1 { 1.2pt + t.ink } else { 0.75pt + t.muted.lighten(55%) },
      bottom: 0.75pt + t.muted.lighten(55%),
    ),
    inset: (x: 0.45cm, y: 0.5cm),
  )
  show table: set text(size: 0.85em)
  show table: set par(justify: false)
  show table.cell.where(y: 0): set text(font: t.head-font, weight: "bold")
  show figure.where(kind: table): set figure.caption(position: top)

  // subheadings inside sections (## in the qmd)
  show heading: it => block(above: 1em, below: 0.5em,
    if t.head-font == t.body-font { text(size: 34pt, style: "italic", it.body) }
    else { text(font: t.head-font, size: 32pt, weight: "bold", it.body) })

  // figures
  set figure(gap: 0.6cm)
  show figure: set block(above: 0.9cm, below: 0.9cm)
  show figure.caption: it => align(left, text(size: 0.8em, fill: t.muted)[
    #text(weight: "bold", fill: t.ink)[#it.supplement #context it.counter.display(it.numbering).]
    #it.body
  ])

  grid(
    rows: (auto, 1fr, auto),
    poster-header(t, title, subtitle, event, authors, affiliations, logo),
    pad(x: 3cm, top: 2cm, bottom: 1cm, doc),
    poster-footer(t, footer-logos, funding, contact),
  )
}
