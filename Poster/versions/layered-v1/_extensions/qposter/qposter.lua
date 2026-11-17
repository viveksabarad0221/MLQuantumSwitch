-- QPoster: map fenced divs in poster.qmd to Typst layout functions.
--
-- Narrative structure
--   ::: {.lead}                                   -> one-sentence summary under the header
--   ::: {.layer num="01" label="Question" title="..."}  -> a layer hanging off the spine
--   ::: {.bridge}                                 -> transition sentence between layers (arrow on spine)
--
-- Inside layers
--   :::: {.poster-grid widths="1fr, 1fr, 1fr"}   -> columns; each child is ::: {.col}
--   ::: {.panel title="..."}                      -> titled panel
--   :::: {.flow}                                  -> boxes joined by arrows; children ::: {.col}
--                                                    (add .hl to a .col to shade it)
--   ::: {.keybox}  ::: {.note}  ::: {.placeholder height="15cm"}  ::: {.smallrefs}
--   [text]{.ab} / [text]{.ba}                     -> colour text by causal order
--
-- Titles are passed as Typst markup: keep them plain text.

if not quarto.doc.is_format("typst") then
  return {}
end

local function raw(s) return pandoc.RawBlock("typst", s) end

local function wrap(open, blocks, close)
  local out = { raw(open) }
  for _, b in ipairs(blocks) do table.insert(out, b) end
  table.insert(out, raw(close))
  return out
end

local function columns(el, fn)
  local out = { raw(fn) }
  for _, b in ipairs(el.content) do
    if b.t == "Div" and b.classes:includes("col") then
      table.insert(out, raw(b.classes:includes("hl") and "hl[" or "["))
      for _, c in ipairs(b.content) do table.insert(out, c) end
      table.insert(out, raw("],"))
    end
  end
  table.insert(out, raw(")"))
  return out
end

local function str(s) return '"' .. (s or ""):gsub('"', '\\"') .. '"' end

function Span(el)
  for _, cls in ipairs({ "ab", "ba" }) do
    if el.classes:includes(cls) then
      local out = { pandoc.RawInline("typst", "#" .. cls .. "[") }
      for _, c in ipairs(el.content) do table.insert(out, c) end
      table.insert(out, pandoc.RawInline("typst", "]"))
      return out
    end
  end
end

function Div(el)
  local a = el.attributes
  local c = el.classes
  if c:includes("poster-grid") then
    return columns(el, "#poster-grid(widths: (" .. (a.widths or "1fr, 1fr, 1fr") .. "),")
  elseif c:includes("twocol") then
    return columns(el, "#twocol(widths: (" .. (a.widths or "1fr, 1fr") .. "),")
  elseif c:includes("flow") then
    local w = a.widths and ("widths: (" .. a.widths .. "),") or ""
    return columns(el, "#flow(" .. w)
  elseif c:includes("layer") then
    return wrap("#layer(num: " .. str(a.num) .. ", label: [" .. (a.label or "") ..
      "], title: [" .. (a.title or "") .. "])[", el.content, "]")
  elseif c:includes("bridge") then
    return wrap("#bridge[", el.content, "]")
  elseif c:includes("lead") then
    return wrap("#lead[", el.content, "]")
  elseif c:includes("panel") then
    return wrap("#panel(title: [" .. (a.title or "") .. "])[", el.content, "]")
  elseif c:includes("card") then
    local numbered = (a.numbered == "false") and "false" or "true"
    return wrap("#pcard(title: [" .. (a.title or "") .. "], numbered: " .. numbered .. ")[",
      el.content, "]")
  elseif c:includes("keybox") then
    return wrap("#keybox[", el.content, "]")
  elseif c:includes("note") then
    return wrap("#note[", el.content, "]")
  elseif c:includes("smallrefs") then
    return wrap("#text(size: 0.75em)[", el.content, "]")
  elseif c:includes("placeholder") then
    return wrap("#placeholder(height: " .. (a.height or "15cm") .. ")[", el.content, "]")
  end
end
