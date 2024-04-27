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
<div>
    <p>Cut the Image <b> Als erstes wird das Image auf eine fest größe Zugeschnitten </p>
    <p>Rotate the Image - Danach wird das Image random Rotiert</p>
    <p>Noise the Image - Zum schluss wird das Image per Trapez Form randomized und die weißen Pixel werden zu einer Papier ähnlichen Farbe genoised</p>
    <p>Label of the Image - Hier sieht man die ground Truths der Images</p>
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
