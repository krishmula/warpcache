# Social preview

`social-preview.png` is the GitHub social preview image: 1280 × 640 pixels,
with an opaque background and a file size under 1 MB. The editable vector
source is `social-preview.svg`.

To upload it, open the repository's **Settings → General → Social preview →
Edit → Upload an image**, then select `social-preview.png`.
Committing the image alone does not set the repository's social preview.

GitHub's image requirements and upload instructions are in the
[social preview documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview).

To render the PNG again from the repository root, with Cairo installed:

```sh
uv run --no-project --with cairosvg python -c \
  "import cairosvg; cairosvg.svg2png(url='docs/assets/social-preview.svg', write_to='docs/assets/social-preview.png')"
```

On macOS with Homebrew Cairo, prefix the command with
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` if Cairo cannot be found.
