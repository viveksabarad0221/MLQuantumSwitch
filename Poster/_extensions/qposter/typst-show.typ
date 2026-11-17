#show: doc => poster(
$if(title)$
  title: [$title$],
$endif$
$if(subtitle)$
  subtitle: [$subtitle$],
$endif$
$if(event)$
  event: "$event$",
$endif$
  authors: (
$for(poster-authors)$
    (name: [$it.name$], aff: [$it.aff$]),
$endfor$
  ),
  affiliations: (
$for(poster-affiliations)$
    [$poster-affiliations$],
$endfor$
  ),
$if(poster-logo)$
  logo: "$poster-logo$",
$endif$
  footer-logos: (
$for(footer-logos)$
    "$footer-logos$",
$endfor$
  ),
$if(poster-funding)$
  funding: [$poster-funding$],
$endif$
$if(contact)$
  contact: [$contact$],
$endif$
$if(poster-theme)$
  theme: "$poster-theme$",
$endif$
$if(fontsize)$
  fontsize: $fontsize$,
$endif$
  doc,
)
