# TraStati

Generierung und erkennung statischer Systeme mittels Transformer

# Usage

### Generierung eines Datensets

```python
python main --data
```

<div>
    <img src="assets/cut_image.jpg" width="200" alt="Cut Image">
    <img src="assets/rotated_image.jpg" width="200" alt="Rotated Image"> 
    <img src="assets/noised_image.jpg" width="200" alt="Noised Image">
    <img src="assets/output_image.jpg" width="200" alt="Output Image">

</div>

<div style="max-width: 300px;">
  <img src="assets/noised_image.jpg" alt="Image Description" style="max-width: 100%; height: auto;">
  <p style="max-width: 100%; overflow-wrap: break-word;">Description of the image. This text will wrap to the next line if it exceeds the width of the image.</p>
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
