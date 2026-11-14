"""Mechanical extraction to share ownership/update logic with a jerk state."""
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'rotor_force_observer.py'
s=p.read_text('utf-8');original=s
start=s.index('            F = np.eye(7);Q = np.zeros((7, 7))')
end=s.index('            self.x = F@self.x;',start)
block=s[start:end]
function='\n    def _transition(self,dt):\n'+''.join(line[4:]+'\n' for line in block.splitlines())+'        return F,Q\n'
s=s[:start]+'            F,Q=self._transition(dt)\n'+s[end:]
initial='''            self.x = np.r_[z, np.zeros(4), self.default_gain]
            self.P = np.diag([10000.]*4+[1000000.]*2+[self.default_gain**2])
            self.P[2:4,2:4]=np.eye(2)*self.velocity_sigma**2
'''
assert initial in s
s=s.replace(initial,'            self._initialize(z)\n',1)
function+='''
    def _initialize(self,z):
        self.x = np.r_[z, np.zeros(4), self.default_gain]
        self.P = np.diag([10000.]*4+[1000000.]*2+[self.default_gain**2])
        self.P[2:4,2:4]=np.eye(2)*self.velocity_sigma**2
'''
s+=function
s=s.replace('    correlation_s = .35','    state_dim=7\n    gain_index=6\n    correlation_s = .35',1)
replacements={
 'np.eye(7)':'np.eye(self.state_dim)',
 'np.zeros((7, 7))':'np.zeros((self.state_dim,self.state_dim))',
 'np.zeros((7,7))':'np.zeros((self.state_dim,self.state_dim))',
 'np.zeros((7, 2))':'np.zeros((self.state_dim,2))',
 'np.zeros((2, 7))':'np.zeros((2,self.state_dim))',
 'np.zeros((4,7))':'np.zeros((4,self.state_dim))',
 'self.x[6]':'self.x[self.gain_index]',
 'self.P[6, 6]':'self.P[self.gain_index,self.gain_index]',
 'self.P[6,6]':'self.P[self.gain_index,self.gain_index]',
 'J[4:6, 6]':'J[4:6,self.gain_index]',
 'J[4:6,6]':'J[4:6,self.gain_index]',
 'F[:2,6]':'F[:2,self.gain_index]',
 'F[2:4,6]':'F[2:4,self.gain_index]',
 'F[4:6,6]':'F[4:6,self.gain_index]',
 'F[:2, 6]':'F[:2,self.gain_index]',
 'F[2:4, 6]':'F[2:4,self.gain_index]',
 'H[:,6]':'H[:,self.gain_index]',
 'K[6]':'K[self.gain_index]',
 'delta[6]':'delta[self.gain_index]'}
for old,new in replacements.items():s=s.replace(old,new)
assert s!=original
Path(__file__).resolve().parents[2].joinpath('backup/20261104_pre_native165/rotor_force_observer.py').write_text(original,'utf-8')
p.write_text(s,'utf-8')
