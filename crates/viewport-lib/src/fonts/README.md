# Built-in font

`Roboto-Regular.ttf` is Roboto Regular v2.138 (Apache 2.0, see `Roboto-LICENSE.txt`), from the `roboto-unhinted.zip` release at https://github.com/googlefonts/roboto/releases/tag/v2.138, cut down to keep the embedded size small.

It keeps these characters, with hinting removed (the overlay renderer does not hint glyphs) and the default OpenType layout features:

| Range | Block |
| --- | --- |
| U+0020-007E, U+00A0-024F | Basic Latin, Latin-1, Latin Extended-A and -B |
| U+0370-03FF | Greek |
| U+0400-04FF | Cyrillic |
| U+2000-206F | General Punctuation |
| U+2070-209F | Superscripts and Subscripts |
| U+20A0-20CF | Currency Symbols |
| U+2100-214F | Letterlike Symbols |
| U+2190-21FF | Arrows |
| U+2200-22FF | Mathematical Operators |

Text outside this set renders as missing glyphs; pass a font that covers it to `upload_font`.

To rebuild it (`pyftsubset` is part of fontTools):

```bash
pyftsubset Roboto-Regular.ttf --no-hinting --output-file=Roboto-Regular.subset.ttf \
  --unicodes="U+0020-007E,U+00A0-024F,U+0370-03FF,U+0400-04FF,U+2000-206F,U+2070-209F,U+20A0-20CF,U+2100-214F,U+2190-21FF,U+2200-22FF"
```
