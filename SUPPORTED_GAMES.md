# Supported games

The following DirectX 12 games run DLSS Super Resolution through d4r on the tested Radeon RX 7700 XT Linux setup with GE-Proton 11-3. All three documented DLSS models run in each game; image-quality issues are noted in the table.

| Game | E: DLSS 3 CNN | K: DLSS 4 | M: DLSS 4.5 |
|---|---|---|---|
| SILENT HILL Townfall | Works | Works | Works |
| Ghost of Tsushima DIRECTOR'S CUT | Works | Works | Works |
| Ready or Not | Works | Works; visible graphical artifacts | Works; minor artifacts |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

Choose a model with `Model = E`, `Model = K`, or `Model = M` under `[DLSS]` in the game's `d4r/d4r.ini`. Model compatibility does not imply equal performance; see the [Townfall measurements](docs/performance.md) for measured frame rates.
