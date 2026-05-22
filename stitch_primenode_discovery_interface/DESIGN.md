---
name: PrimeNode UI
colors:
  surface: '#141219'
  surface-dim: '#141219'
  surface-bright: '#3a3840'
  surface-container-lowest: '#0f0d14'
  surface-container-low: '#1c1b21'
  surface-container: '#211f26'
  surface-container-high: '#2b2930'
  surface-container-highest: '#36343b'
  on-surface: '#e6e0ea'
  on-surface-variant: '#cac4d4'
  inverse-surface: '#e6e0ea'
  inverse-on-surface: '#322f37'
  outline: '#948e9e'
  outline-variant: '#494552'
  surface-tint: '#cdbdff'
  primary: '#cdbdff'
  on-primary: '#361287'
  primary-container: '#a48afb'
  on-primary-container: '#39178a'
  inverse-primary: '#654bb8'
  secondary: '#e8bea2'
  on-secondary: '#442a17'
  secondary-container: '#5e402b'
  on-secondary-container: '#d6ad92'
  tertiary: '#96d3bd'
  on-tertiary: '#00382b'
  tertiary-container: '#6ba792'
  on-tertiary-container: '#003b2d'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e8deff'
  primary-fixed-dim: '#cdbdff'
  on-primary-fixed: '#20005f'
  on-primary-fixed-variant: '#4d319e'
  secondary-fixed: '#ffdcc5'
  secondary-fixed-dim: '#e8bea2'
  on-secondary-fixed: '#2c1605'
  on-secondary-fixed-variant: '#5e402b'
  tertiary-fixed: '#b1efd8'
  tertiary-fixed-dim: '#96d3bd'
  on-tertiary-fixed: '#002118'
  on-tertiary-fixed-variant: '#0d503f'
  background: '#141219'
  on-background: '#e6e0ea'
  surface-variant: '#36343b'
typography:
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 40px
    fontWeight: '600'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 28px
    fontWeight: '500'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Roboto Flex
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Roboto Flex
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  data-mono:
    fontFamily: Roboto Flex
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.05em
  label-sm:
    fontFamily: Roboto Flex
    fontSize: 11px
    fontWeight: '700'
    lineHeight: '1'
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 12px
  md: 24px
  lg: 32px
  xl: 48px
  layout_split: 30% 70%
---

## Brand & Style
The design system is centered on the concept of "Analytical Precision"—a digital translation of high-end laboratory instrumentation. It balances the stark, information-dense requirements of scientific research with the fluid, approachable geometry of modern Material design. 

The aesthetic is **High-Tech Laboratory**, evoking the feeling of a specialized hardware interface. It utilizes a dark-mode first strategy to reduce eye strain during prolonged data analysis. The emotional response is one of expert control, high confidence, and futuristic utility. It blends **Minimalism** with **Modern Corporate** structure, ensuring that while the interface feels advanced, it remains highly functional and predictable for technical users.

## Colors
The palette is built on a "Deep Sea" foundation to provide maximum contrast for technical data. 

- **Primary Action (Lavender):** Reserved for core interactions, active states, and primary navigational paths.
- **High Confidence / Match (Mint Green):** A semantic indicator used for positive correlations, high-probability data matches, and "Success" states.
- **Secondary / Investigation (Soft Orange):** Used for elements requiring user scrutiny, warnings, or secondary data points that sit outside the primary "Confidence" path.
- **Surface Strategy:** Layers are built using tonal shifts rather than opacity. The base (#0F1115) acts as the "instrument chassis," while the container (#1A1C23) represents the "active modules" or cards.

## Typography
The typographic hierarchy distinguishes between "Context" and "Content." 

**Space Grotesk** is used for headlines and structural landmarks, providing a geometric, technical character that mirrors the instrument aesthetic. 

**Roboto Flex** handles the heavy lifting for data density. As a variable font, it is optimized for legibility in tight spaces. Data points and technical labels should utilize the `data-mono` and `label-sm` styles to maintain a rigorous, systematic feel. Use higher font weights for data values and lighter weights for data units to create immediate visual parsing.

## Layout & Spacing
The layout follows an **Asymmetric Split** model (30/70). The left 30% is dedicated to "Parameter Control" and "Search Nodes," while the right 70% serves as the "Primary Observation Deck" for data visualization and results.

A 12-column fluid grid is used within each split section. Gutters are kept tight (12px to 16px) to maintain the "dense data" laboratory look, while outer margins are generous (32px to 48px) to frame the interface as a premium tool. Content density should be high within cards, utilizing a 4px baseline grid for internal element alignment.

## Elevation & Depth
In this design system, depth is communicated through **Tonal Stacking** and **Subtle Outlines** rather than traditional drop shadows.

- **Level 0 (Base):** #0F1115 (The background).
- **Level 1 (Module):** #1A1C23 (The card/container).
- **Interactive States:** Use a 1px solid border in the Primary or Secondary color to indicate focus. 
- **Backdrop Blurs:** High-priority overlays (modals) should use a 20px blur with a 40% opacity fill of the base color to maintain the "glass instrument" feel.
- **Indicators:** Use soft outer glows (bloom) of 4-8px for High Confidence states to make them appear "illuminated" from within the display.

## Shapes
The shape language is characterized by **Extremes**. 

Structural containers like cards, main inputs, and large buttons use a **24px radius** (Level 3/Pill-shaped). This creates a distinct contrast against the dense, rigid technical data inside them. Smaller interactive elements like chips, tags, and "Match Score" segments should also follow this hyper-rounded philosophy to ensure a cohesive Material 3 visual flow. 

Data visualization nodes (MoA breadcrumbs) are perfect circles to emphasize their status as individual data points within a network.

## Components
- **Target Node Search:** A pill-shaped, high-height input field with a persistent search icon. It uses the Surface Container color with a Primary Action border on focus.
- **Match Score Rings:** Concentric, segmented progress rings that use the High Confidence (Mint) for positive values and the Secondary (Orange) for anomalies.
- **MoA Visual Breadcrumbs:** Circular nodes connected by thin (1px) lavender lines, representing the "Mechanism of Action" path. Active nodes have a subtle primary-color bloom.
- **Cards:** Use the 24px corner radius. Header areas within cards should be separated by a subtle 1px divider in a slightly lighter shade of the surface color.
- **Buttons:** All buttons are pill-shaped. Primary buttons use a solid Lavender fill with dark text. Secondary buttons use a ghost style with a 1.5px Lavender border.
- **Data Tables:** Dense layout with no vertical borders. Row hover states use a subtle lightening of the surface container.