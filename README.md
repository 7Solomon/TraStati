# TraStati

Generierung und erkennung statischer Systeme mittels Transformer

# Usage

### Generierung eines Datensets

```python
python main --data
```

<style>
  .image-container {
    display: flex; /* Setzt die Elemente als Flexbox */
    flex-wrap: wrap; /* Erlaubt das Umwickeln der Elemente in die nächste Zeile */
  }

  .image-container img {
    width: 200px; /* Setzt die Breite der Bilder */
    margin: 5px; /* Setzt den Abstand zwischen den Bildern */
  }

  .image-container p {
    width: 200px; /* Setzt die maximale Breite des Texts */
    margin: 5px; /* Setzt den Abstand zwischen den Bildern */
  }
</style>

<div class="image-container">
  <img src="assets/cut_image.jpg" alt="Cut Image">
  <p>Cut the Image - Als erstes wird das Image auf eine feste Größe zugeschnitten</p>

  <img src="assets/rotated_image.jpg" alt="Rotated Image">
  <p>Rotate the Image - Danach wird das Image zufällig rotiert</p>

  <img src="assets/noised_image.jpg" alt="Noised Image">
  <p>Noise the Image - Zum Schluss wird das Image per Trapezform randomisiert und die weißen Pixel werden in eine papierähnliche Farbe geräuschbehaftet</p>

  <img src="assets/output_image.jpg" alt="Output Image">
  <p>Label of the Image - Hier sieht man die Ground Truths der Images</p>
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
