import pyvisa

class DG1022:
    def __init__(self):
        self.phase = 0
        self.rm = pyvisa.ResourceManager()
        
        for res in rm.list_resources():
            if "DG1" in res:
                self.inst = rm.open_resource(res)
                break
                
        self.inst.write("SOUR1:FUNC SIN")
    
    def start(self, freq, amp):
        self.inst.write(f"SOUR1:FREQ {freq}")
        self.inst.write(f"SOUR1:VOLT {amp}")
        self.inst.write("OUTP1 ON")

    def flip(self):
        if self.phase == 0: self.inst.write("SOUR1:PHAS 180"); self.phase = 180
        else: self.inst.write("SOUR1:PHAS 0"); self.phase = 0

    def stop(self):
        self.inst.write("OUTP1 OFF")

    def close(self):
        self.inst.close()
