# quantum-triplet-loss

### Tests mit positivem Effekt
- ZZ Measurement auf Qubits 0,1 und 2,3

### Tests mit keinem Effekt
- StronglyEntanglingLayers
- Veränderung von Qubit- und Layer-Anzahl
- Veränderung der LR (dynamisch und statisch) - 0,01 führt zu bestem Ergebnis
- Veränderung des Alpha-Wertes 
    - Je größer umso größer die Varianz der Messung
    - Veränderung des Alpha-Wertes macht vorherige Steps überflüssig
- 3 Qubits zum Auslesen
- Ein separates Circuit pro Messung
- Data-Reuploading

### Tests mit negativem Effekt
- Online Mining der Triplets (alles wird auf ein Wert abgebildet)
- Erst normales Training, dann Online Mining
- Jegliche Veränderung des Loss