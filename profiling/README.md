## Acoustic injection experiment system for the Raspberry Pi

### Components
* BMI160 6-DoF IMU
* AD9833 Signal Generator
* DS3502 Digital Potentiometer
* TDA8932 Mono Amplifier
* USB Audio Interface
* PZT Disc
* SG90 Servo

## Wiring Guide
### BMI160:
| A | B |
|-|-|
| VIN | Pi 3.3v |
| GND | Pi GND |
| SCL | Pi SCL |
| SDA | Pi SDA |
| SAO | Pi GND |

### AD9833:
| A | B |
|-|-|
| VCC | Pi 3.3v |
| DGND | Pi GND |
| SDATA | Pi MOSI |
| SCLK | Pi SCLK |
| FSYNC | Pi CE0 |
| AGND | Pi GND |
| OUT | DS3502 RH |

### DS3502:
| A | B |
|-|-|
| V+ | Pi 5v |
| VCC | Pi 3.3v |
| GND | Pi GND |
| SCL | Pi SCL |
| SDA | Pi SDA |
| RL | Pi GND |
| RW | TDA8932 IN+ |
| RH | AD9833 OUT |

### TDA8932:
| A | B |
|-|-|
| IN+ | DS3502 RW |
| IN- | Pi GND |
| OUT +/- | PZT Disc |
| PWR +/- | DC PSU (24V, 2A) |

### SG90:
| A | B |
| - | - |
| Signal | Pi GPIO 18 |
| Power | Pi 5v |
| Ground | Pi GND |
