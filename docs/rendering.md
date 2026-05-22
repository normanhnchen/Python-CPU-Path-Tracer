## Rendering

Important notes on the output render of this path tracer.

## Color Output

- **Float RGB**: (0-1 decimal values)
- **Int RGB**: (0-255 integer values)

The path tracer outputs total accumulated lighting in floats. We need to clip this output to 0-1  
values to be valid float RGB color data.

**Intel Open Image Denoise** requires float RGB input, so denoising is performed before  
converting to int RGB.

**Pygame** requires int RGB input for display, therefore we convert from float RGB to int RGB by  
multiplying by 255.

## Tonemapping

The path tracer outputs color in high dynamic range (HDR) data, because it outputs actual light  
intensity. However, the brightness and contrast range of HDR is far too wide for a standard  
display. Therefore, converting to a lower dynamic range (LDR) is required for proper lighting.

**Tonemapping** is the process of converting high dynamic range (HDR) into lower dynamic  
range (LDR), adjusting brightness and contrast to be more realistic.

## Gamma Correction

The path tracer outputs color in linear RGB which is actual light intensity. However, human eyes  
perceive brightness non-linearly.

To account for this images are gamma-corrected to a gamma curve of about 1 / 2.2.

## Pipeline

<center>

```
Path Tracer Output (HDR)
↓
Clip values to 0–1 (float RGB)
↓
Denoise (float RGB)
↓
Tonemap HDR → LDR (float RGB)
↓
Gamma Correction (float RGB)
↓
Convert float RGB → int RGB
↓
Display using Pygame
```

</center>

## References

https://learnopengl.com/Advanced-Lighting/Gamma-Correction