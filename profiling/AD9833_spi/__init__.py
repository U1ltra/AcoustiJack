import spidev

class AD9833:
    def __init__(self):
        self.spi = spidev.SpiDev()

        self.spi.open(0, 0)
        self.spi.max_speed_hz = 12_500_000  # 12.5 MHz
        self.spi.mode = 2

        self.CLK = 25_000_000

        self.phase = 0

        self.write = lambda word: self.spi.xfer2([(word >> 8) & 0xFF, word & 0xFF])
    
    def start(self, freq):
        word = int(round(freq, 1) * (2**28) / self.CLK)

        LOW14 = word & 0x3FFF
        HIGH14 = (word >> 14) & 0x3FFF

        self.write(0x2100)
        self.write(0x4000 | LOW14)
        self.write(0x4000 | HIGH14)
        self.write(0x2000)

    def flip(self):
        if self.phase == 0: self.write(0xC000 | 2048); self.phase = 2048
        else: self.write(0xC000 | 0); self.phase = 0  

    def stop(self):
        self.write(0x2100 | 0x100)

    def close(self):
        self.spi.close()
