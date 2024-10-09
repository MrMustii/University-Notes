1.  trichromic coordinates 
2. CIECMF1931 constant tabs (xyz)
3. uv1960 (not used) calculates the CIE 1960 UCS coordinates of the given spectrum 
4. CCT (correlated color temperature) 
5. MEDI calculates the MacAdam, ellipse index (MEDI) of a given spectrum 
6. illuminance calculates the  in lux
7. spectrem calculation calculates the spectrum
	1. 2 correction vectors "to calculate normalized data of the sensor" 
	2. import correction matrix file 
	3. caluclulate the real gain
	4. calculate the intergrated step
	5. do $$BasicCount=\frac{channel Value}{inter*realGain}$$
	6. do $$correctedData=factorCorr*(basicCount-offsetCorr)$$
	7. $$specter=CorrectionMatrixXcorrectedData^T$$
	