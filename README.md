# TraStati

Generierung und erkennung statischer Systeme mittels Transformer

# Usage

### Generierung eines Datensets

```python
python main --data
```

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <div style="width: 200px; margin-right: 10px;">
    <img src="assets/cut_image.jpg" alt="Cut Image" style="width: 100%;">
    <p>Cut the Image - Als erstes wird das Image auf eine feste Größe zugeschnitten.</p>
  </div>
  
  <div style="width: 200px; margin-right: 10px;">
    <img src="assets/rotated_image.jpg" alt="Rotated Image" style="width: 100%;">
    <p>Rotate the Image - Danach wird das Image zufällig rotiert.</p>
  </div>
  
  <div style="width: 200px; margin-right: 10px;">
    <img src="assets/noised_image.jpg" alt="Noised Image" style="width: 100%;">
    <p>Noise the Image - Zum Schluss wird das Image per Trapezform randomisiert und die weißen Pixel werden in eine papierähnliche Farbe geräuschbehaftet.</p>
  </div>
  
  <div style="width: 200px;">
    <img src="assets/output_image.jpg" alt="Output Image" style="width: 100%;">
    <p>Label of the Image - Hier sieht man die Ground Truths der Images.</p>
  </div>
</div>

### Visualisierung eines Datasets

```python
python main --display
```

### Visualisierung des outputs eines Models

```python
python main --test
```

### Trainieren eines Models

```python
python main --train
```

# Installation

```python
pip install -r requirements.txt
```

Da die Daten mit "pdflatex" mit python subprocess generiert werden, müssen pdflatex, convert installiert sein.

## Other dependencies

### Linux (Debian)

```console
$ sudo apt install texlive
$ sudo apt install imagemagick

```
