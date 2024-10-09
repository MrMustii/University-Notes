#### CIE 1931
- Links between distributions of wavelengths in the electromagnetic visible spectrum, and physiologically perceived colors in human color vision
- Weighting a total light power spectrum by the individual spectral sensitivities of the three kinds of cone cells renders three effective values of [stimulus](https://en.wikipedia.org/wiki/Stimulus_(physiology) "Stimulus (physiology)"); these three values compose a tristimulus specification of the objective color of the light spectrum.The three parameters, denoted "S", "M", and "L", are indicated using a [3-dimensional](https://en.wikipedia.org/wiki/3-dimensional "3-dimensional") space denominated the "[LMS color space](https://en.wikipedia.org/wiki/LMS_color_space "LMS color space")", which is one of many color spaces devised to quantify human [color vision](https://en.wikipedia.org/wiki/Color_vision "Color vision").
- The CIE XYZ color space encompasses all color sensations that are visible to a person with average eyesight. That is why CIE XYZ (Tristimulus values) is a device-invariant representation of color.[[5](https://en.wikipedia.org/wiki/CIE_1931_color_space#cite_note-5)] 
- Certain tristimulus values are thus physically impossible: e.g. LMS tristimulus values that are non-zero for the M component and zero for both the L and S components.![[Pasted image 20240331165451.png]]
- In XYZ space, all combinations of non-negative coordinates are meaningful, but many, such as the primary locations [1, 0, 0], [0, 1, 0], and [0, 0, 1], correspond to [imaginary colors](https://en.wikipedia.org/wiki/Imaginary_color "Imaginary color") outside the space of possible LMS coordinates; imaginary colors do not correspond to any spectral distribution of wavelengths and therefore have no physical reality.
- In the CIE 1931 model, _Y_ is the [luminance](https://en.wikipedia.org/wiki/Luminance "Luminance"), _Z_ is quasi-equal to blue (of CIE RGB), and _X_ is a mix of the three CIE RGB curves chosen to be nonnegative (see [§ Definition of the CIE XYZ color space](https://en.wikipedia.org/wiki/CIE_1931_color_space#Definition_of_the_CIE_XYZ_color_space))



- The concept of color can be divided into two parts: brightness and [chromaticity](https://en.wikipedia.org/wiki/Chromaticity "Chromaticity"). For example, the color white is a bright color, while the color grey is considered to be a less bright version of that same white. In other words, the chromaticity of white and grey are the same while their brightness differs.
- The CIE XYZ color space was deliberately designed so that the _Y_ parameter is also a measure of the [luminance](https://en.wikipedia.org/wiki/Luminance "Luminance") of a color. The chromaticity is then specified by the two derived parameters _x_ and _y_, two of the three normalized values being functions of all three [tristimulus values](https://en.wikipedia.org/wiki/CIE_1931_color_space#Tristimulus_values) _X_, _Y_, and _Z_
$$x = \frac{X}{X + Y + Z}$$
$$y = \frac{Y}{X + Y + Z}$$
$$z = \frac{Z}{X + Y + Z} = 1 - x - y$$
- ![[Pasted image 20240331173340.png]]
- The outer curved boundary is the _spectral locus_, with wavelengths shown in nanometers. The chromaticity diagram is a tool to specify how the human eye will experience light with a given spectrum. It cannot specify colors of objects
- 




#### Questions to ask
- Why not use CIE 1976 UV color space
- Still not sure what the file is all about or what the xbar is or how to get X
----

#### MacAdam ellipse
-  Is roughly a region on a [chromaticity diagram](https://en.wikipedia.org/wiki/Chromaticity_diagram "Chromaticity diagram") which contains all colors which are indistinguishable, to the average human eye, from the color at the center of the ellipse
- 