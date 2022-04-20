import numpy as np
np.random.seed(0)


ngratings = 10
nwaves = 3
gratings = np.random.randn(nwaves, ngratings, 2, 2)

waves = []
for wave in gratings:
    out = np.einsum('ijk,jmn,->', wave, wave, wave)
    waves.append(out)
print(out)

print("____________________________________________________")
waves = []
for i in range(nwaves):
    wave = np.eye(2)
    for grat in gratings[i, :, :, :]:
        wave = wave @ grat 
    waves.append(wave)
waves = np.array(waves)

print(waves)
