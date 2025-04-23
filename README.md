# Analyse der indirekten wirtschaftlichen Effekte von Stadionbesuchen in der MEWA ARENA

Dieses Repository enthält das Python-Skript `20241210_SUF_code.py`, das zur Auswertung von Umfragedaten aus dem Stadionumfeld des 1. FSV Mainz 05 e.V. verwendet wird. 
Ziel ist es, indirekte wirtschaftliche Effekte durch Ausgaben von Besucherinnen und Besuchern zu quantifizieren.

## Zweck des Codes

Das Skript dient der Verarbeitung, Bereinigung und Analyse von Umfragedaten, die im Stadion erhoben wurden. Es umfasst:

- Datenimport und -bereinigung
- Geodatenverarbeitung (Entfernungsberechnung, PLZ-Zuordnung)
- Erstellung interaktiver Karten und Choroplethen
- Erstellung von Mittelwertvariablen aus Kategorien
- Durchführung ordinaler und multinomialer logistischer Regressionsanalysen
- Berechnung durchschnittlicher und saisonal hochgerechneter Ausgaben
- Durchführung von Sensitivitätsanalysen und Modellvergleichen (z. B. AIC, Log-Likelihood)

## Datengrundlage

Die verwendeten Daten stammen aus einer Stadionumfrage im Zeitraum Oktober bis Dezember 2024, die mit Unterstützung der Johannes Gutenberg-Universität Mainz durchgeführt wurde. 
Die Befragung erfasste Informationen zu Demografie, Anreise, Übernachtung, Verpflegung, Freizeitverhalten und Motivation der Stadionbesucherinnen und -besucher. 

Hier geht es zur Umfrage (eventuell nicht mehr aktiv):  
https://survey.zdv.uni-mainz.de/index.php/444469?lang=de

### Hochgeladene Dateien im Repository

- `umfrage_dates.xlsx` – Liste der Umfrageerhebungstage

### Öffentlich verfügbare Datein
Die folgenden Datendateien sind öffentlich zugänglichen oder generischen Quellen stammen:

- `postleitzahlen.csv` – Postleitzahlen in Deutschland inkl. Koordinaten und Kreisinformation: https://www.govdata.de/suche/daten/deutschland-postleitzahlen  
- `georef-germany-kreis.geojson` – Geodaten der deutschen Landkreise für Kartendarstellungen: https://public.opendatasoft.com/explore/dataset/georef-germany-kreis/export/?


### Nicht enthaltene Dateien

Die folgenden Dateien werden im Skript verwendet, aber **nicht im Repository bereitgestellt**:

- `final.csv` – Umfragedatensatz  
  *Nicht enthalten, da die Daten personenbezogen bzw. pseudonymisiert sind und ausschließlich lokal an der Universität Mainz im Einklang mit den Datenschutzvorgaben verarbeitet werden.*

- `ticketverkauf.xlsx` – interne Daten zu Ticketverkäufen des Vereins  
  *Nicht enthalten, da diese Daten vertrauliche Informationen des Vereins enthalten.*

## Nutzungshinweise

- Vor der Ausführung sind ggf. lokale Pfade im Skript anzupassen.
- Die Ausführung setzt das lokale Vorliegen der oben genannten (nicht enthaltenen) Dateien voraus.
- Das Skript wurde mit Python 3.11.9 durchgeführt.

## Autorin

Ella Jurk  
IPE – Institut for Policy Evaluation  
E-Mail: e.jurk@ipe-evaluation.de

© 2024 IPE Institut für Politikevaluation GmbH. Alle Rechte vorbehalten.
