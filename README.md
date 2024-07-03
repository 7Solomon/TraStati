# TraStati

Generierung und erkennung statischer Systeme mittels Transformer

# Usage

### Generierung eines Datensets

```python
python main --data
```

### 3 stufige Generierung

<div>
    <img src="assets/cut_image.jpg" width="200" alt="Cut Image">
    <img src="assets/rotated_image.jpg" width="200" alt="Rotated Image"> 
    <img src="assets/noised_image.jpg" width="200" alt="Noised Image">
</div>

1. Cut the Image - Als erstes wird das Image auf eine fest größe Zugeschnitten
2. Rotate the Image - Danach wird das Image random Rotiert
3. Noise the Image - Zum schluss wird das Image per Trapez Form randomized und die weißen Pixel werden zu einer Papier ähnlichen Farbe genoised

### Visualisierung eines Datasets

```python
python main --display
```

<img src="assets/output_image.jpg" width="200" alt="Output Image">
Label of the Image - Hier sieht man die ground Truths der Images

### Visualisierung des outputs eines Models

```python
python main --test
```

##### Output und Heatmaps

Loss von 1.19:

<img src="assets/heatmap_1.jpg" width="300" alt="Heatmap">

Die Predictions folgen keinem Muster, auch die Heatmap sieht sehr random aus.
Auch wieso jeder 2. Head keine Attention hat ist mir noch ein Rätsel.

### Trainieren eines Models

```python
python main --train
```

Der Verlauf der Lossentwicklung nach 5 Epochen mit 109 Batches mit jeweils 6 Images:

<img src="assets/loss_plot_1.jpg" width="300" alt="Loss Plot">

Der Loss des selben model nach weiteren 5 Epochen:

<img src="assets/loss_plot_2.jpg" width="300" alt="Loss Plot">

Wie man sieht springt der Loss sehr und Konvergiert auf keinen Fall gut.

Mögliche Verbesserungen:

- Verändern der Kostenfunktionen.
- Erhöhung der Anzahl an Heads, Encoder/ Decoder layer, sowie Erhöhung der hidden_dim.
(Braucht mehr Leistung, als gerade verfügbar)
- Anpassen der Hyperparameter.


# Installation

Alle pip dependencies können hiermit installiert werden.

```python
pip install -r requirements.txt
```

## Other dependencies

Da die Daten mit "pdflatex" mit python subprocess generiert werden, müssen "pdflatex", "convert" und "texlive-pictures" installiert sein.

#### Linux (Debian)

```console
sudo apt-get install texlive
sudo apt-get install imagemagick
sudo apt-get install texlive-pictures
```

#### Windows ("Theoretische Installation", da noch nicht ausführlich getesetet)

Folge den Anweisungen auf
[TexLive](https://www.tug.org/texlive/windows.html)
um TexLive und TexLive-Pictures zu installieren.

Folge den Anweisungen auf
[ImageMagick](https://imagemagick.org/script/download.php#windows)
um ImageMick zu installieren.


Um PDF in jpgs umzuwandeln wird auf Windows Ghostscsript benötigt.
Folge den Anweisungen auf
[Ghostscript](https://ghostscript.com/releases/gsdnld.html)
um GS zu installieren.


Außerdem müssen die drei programme zum PATH hinzugefügt werden.


##### pip packages auf Windows

Ich hatte bei der installierung auf Windows probleme mit ein paar python packages und unkompatiblen python versions.
Dies muss noch tiefer ausprobiert werden.

Manuel zu installierende Packages:
´´´
torch
...
torchvision
...
opencv-python
matplotlib
latex
numpy<2.0.0

´´´

## path configuration

Da mit custome LaTex librarys gearbeitet wird, muss der path zu dem data_folder configuriert werden.

#### Linux

Dies geht auf Linux mit:

```python
python start_up.py
```

### Windows

Bei Windows muss ausprobiert werden ob das auch funktioniert, oder welche Fehler ausgegeben werden.
Es muss aufjedenfall "src/data_folder/get_system_image" als Library Path für LaTex angegeben werden.

## Google Colab

Hier ist ein Google Collab link, dort kann man einfach den anwesiungen folgen und die Repository mal ausprobieren.

[TraStati Starthilfe](https://colab.research.google.com/drive/1uBNAzD4KOvp-wpPS1W5qIAhtLJrI14kr?usp=sharing)
