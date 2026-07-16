"""
Pure Python reimplementation of TAUOLA Fortran subroutines needed by calculateHH.

Implements (IVER=0, CLEO mode):
  dam2pi, dampry, damppk, dam4pi
and all their dependencies.

Convention: 4-vectors are [px, py, pz, E]  (indices 0,1,2,3)
Minkowski product: p·q = p[3]*q[3] - p[2]*q[2] - p[1]*q[1] - p[0]*q[0]
"""

import math
import cmath

PI = math.pi

# ---------------------------------------------------------------------------
# Physics constants (from PKORB, PARMAS, DECPAR, QEDPRM)
# ---------------------------------------------------------------------------

AMTAU  = 1.777000
AMNUTA = 0.000000
AMEL   = 0.000511
AMNUE  = 0.000000
AMMU   = 0.105658
AMNUMU = 0.000000
AMPIZ  = 0.134976  # pi0
AMPI   = 0.139570  # pi+/-
AMRO   = 0.769900  # rho
GAMRO  = 0.151200
AMA1   = 1.275000  # a1
GAMA1  = 0.700000
AMK    = 0.493677  # K+/-
AMKZ   = 0.497670  # K0
AMKST  = 0.891590  # K*
GAMKST = 0.049800

GFERMI = 1.16637e-5
GV     = 1.0
GA     = -1.0
CCABIB = 0.97492
SCABIB = 0.22252

ALFINV = 137.03604
ALFPI  = ALFINV / PI

# PKORB mixing parameters
_P1_15 = 1.370;   _P2_15 = 0.510;   _P3_15 = -0.110   # rho prime
_P1_16 = 1.700;   _P2_16 = 0.235;   _P3_16 = -0.038   # K* prime
_P1_17 = 1.461;   _P2_17 = 0.250;   _P3_17 =  0.000   # a1 prime
_P1_19 = 1.270;   _P2_19 = 0.090;   _P3_19 =  1.000   # K1A
_P1_20 = 1.402;   _P2_20 = 0.174;   _P3_20 =  0.800   # K1B  (IMIXPP(203)=1 → 0.8)
_P1_21 = 1.465;   _P2_21 = 0.310;   _P3_21 = -0.110   # rho''
_P1_22 = 1.700;   _P2_22 = 0.235;   _P3_22 = -0.110   # rho'''
_P1_14 = 0.781940; _P2_14 = 0.008430                   # omega

# curr_cleo amplitudes / phases / coefficients
_A31 = 0.0;    _A32 = 0.1242; _A33 = 0.1604
_A34 = 0.2711; _A35 = 0.4443; _A36 = 0.0; _A37 = 1.0
_PH42 = -0.40; _PH43 = 0.00;  _PH44 = -0.20 + PI; _PH45 = -1.50
_ALF0v=-0.10; _ALF1v=1.00; _ALF2v=-0.10; _ALF3v=-0.04
_LAM0v=1.00;  _LAM1v=0.14; _LAM2v=-0.05; _LAM3v=-0.05
_BET1v=1.000; _BET2v=-0.145; _BET3v=0.000

# FK1AB amplitudes (derived from PKORB(3,81..88))
def _fk1ab_amps():
    C1270 = _P3_19;  C1402 = _P3_20
    A1270_KSPI = math.sqrt(0.16); A1270_KRHO = math.sqrt(0.42)
    A1402_KSPI = math.sqrt(0.94); A1402_KRHO = math.sqrt(0.03)
    CG1 = -math.sqrt(2./3.);       CG2 = math.sqrt(1./3.)
    return [
        C1270*A1270_KSPI*CG1,  # [0] = PKORB(3,81)
        C1402*A1402_KSPI*CG1,  # [1] = PKORB(3,82)
        C1270*A1270_KRHO*CG1,  # [2] = PKORB(3,83)
        C1402*A1402_KRHO*CG1,  # [3] = PKORB(3,84)
        C1270*A1270_KSPI*CG2,  # [4] = PKORB(3,85)
        C1402*A1402_KSPI*CG2,  # [5] = PKORB(3,86)
        C1270*A1270_KRHO*CG2,  # [6] = PKORB(3,87)
        C1402*A1402_KRHO*CG2,  # [7] = PKORB(3,88)
    ]

_FK1AB_AMPS = _fk1ab_amps()

# COEF table (IVER=0, CLEO) — COEFc(1:5, 0:7), 0-indexed in Python as [j][i-1]
_FPIc = 93.3e-3
_COEF = {
    0: [ 2.*math.sqrt(2.)/3., -2.*math.sqrt(2.)/3.,  2.*math.sqrt(2.)/3., _FPIc,  0.0           ],
    1: [-math.sqrt(2.)/3.,     math.sqrt(2.)/3.,       0.0,                _FPIc,  math.sqrt(2.) ],
    2: [-math.sqrt(2.)/3.,     math.sqrt(2.)/3.,       0.0,                0.0,   -math.sqrt(2.) ],
    3: [ 1./3.,               -2./3.,                  2./3.,              0.0,    0.0            ],
    4: [ 1./math.sqrt(2.)/3., -1./math.sqrt(2.)/3.,    0.0,               0.0,    0.0            ],
    5: [-math.sqrt(2.)/3.,     math.sqrt(2.)/3.,       0.0,               0.0,   -math.sqrt(2.) ],
    6: [ 1./3.,               -2./3.,                  2./3.,             0.0,   -2.0            ],
    7: [ 0.0,                  0.0,                    0.0,               0.0,   -math.sqrt(2./3.)],
}

def coef(i, j):
    """COEF(I,J) — 1-indexed I (1..5), 0-indexed J (0..7)."""
    return _COEF[j][i-1]

# DAMPPK normalization factors
def _fnorm(mnum):
    FPI = _FPIc
    DWAPI0 = math.sqrt(2.0)
    if mnum in (0, 1, 2, 3, 7): return CCABIB / FPI
    if mnum == 4:                return SCABIB / FPI / DWAPI0
    return SCABIB / FPI  # mnum 5, 6

# ---------------------------------------------------------------------------
# Helper: 4-vector Minkowski product
# ---------------------------------------------------------------------------
def _dot4(a, b):
    return a[3]*b[3] - a[2]*b[2] - a[1]*b[1] - a[0]*b[0]

def _mass2(p):
    return p[3]**2 - p[2]**2 - p[1]**2 - p[0]**2

# ---------------------------------------------------------------------------
# BWIG — p-wave BW for rho  (tauola.f:3537)
# ---------------------------------------------------------------------------
def bwig(S, M, G):
    PIM = 0.139
    if S > 4.*PIM**2:
        QS = math.sqrt(abs(S/4. - PIM**2))
        QM = math.sqrt(M**2/4. - PIM**2)
        W  = math.sqrt(S)
        GS = G*(M/W)*(QS/QM)**3
    else:
        GS = 0.0
    return complex(M**2, 0.) / complex(M**2 - S, -M*GS)

# ---------------------------------------------------------------------------
# BWIGM — p-wave BW with running width  (formf.f:244)
# ---------------------------------------------------------------------------
def bwigm(S, M, G, XM1, XM2):
    if S > (XM1+XM2)**2:
        QS = math.sqrt(abs((S      -(XM1+XM2)**2)*(S      -(XM1-XM2)**2))) / math.sqrt(S)
        QM = math.sqrt(abs((M**2   -(XM1+XM2)**2)*(M**2   -(XM1-XM2)**2))) / M
        W  = math.sqrt(S)
        GS = G*(M/W)**2*(QS/QM)**3
    else:
        GS = 0.0
    return complex(M**2, 0.) / complex(M**2 - S, -math.sqrt(S)*GS)

# ---------------------------------------------------------------------------
# BWIGML — L-wave BW  (f3pi.f:253)
# ---------------------------------------------------------------------------
def bwigml(S, M, G, M1, M2, L):
    MP  = (M1+M2)**2
    MM  = (M1-M2)**2
    MSQ = M*M
    W   = math.sqrt(S)
    WGS = 0.0
    if W > (M1+M2):
        QS   = math.sqrt(abs((S   -MP)*(S   -MM))) / W
        QM   = math.sqrt(abs((MSQ -MP)*(MSQ -MM))) / M
        IPOW = 2*L + 1
        WGS  = G*(MSQ/W)*(QS/QM)**IPOW
    return complex(MSQ, 0.) / complex(MSQ - S, -WGS)

# ---------------------------------------------------------------------------
# BWIGS — p-wave BW for K*  (tauola.f:3502)
# ---------------------------------------------------------------------------
def _p_func(A, B, C):
    X = ((A+B-C)**2 - 4.*A*B) / (4.*A)
    return math.sqrt(max(0., X))

def bwigs(S, M, G):
    PIM = 0.139; MK = 0.493667
    PM = _P1_16; PG = _P2_16; PBETA = _P3_16
    QS  = _p_func(S,    PIM**2, MK**2)
    QM  = _p_func(M**2, PIM**2, MK**2)
    W   = math.sqrt(S)
    GS  = G*(M/W)*(QS/QM)**3
    BW  = complex(M**2, 0.) / complex(M**2 - S, -M*GS)
    QPM = _p_func(PM**2, PIM**2, MK**2)
    G1  = PG*(PM/W)*(QS/QPM)**3
    BWP = complex(PM**2, 0.) / complex(PM**2 - S, -PM*G1)
    return (BW + PBETA*BWP) / (1. + PBETA)

# ---------------------------------------------------------------------------
# FPIK — pion form factor  (tauola.f:3562)
# ---------------------------------------------------------------------------
def fpik(W):
    S     = W*W
    ROM   = AMRO;   ROG  = GAMRO
    ROM1  = _P1_15; ROG1 = _P2_15; BETA1 = _P3_15
    return (bwig(S, ROM, ROG) + BETA1*bwig(S, ROM1, ROG1)) / (1. + BETA1)

def fpirho(W):
    return abs(fpik(W))**2

# ---------------------------------------------------------------------------
# FPIKM — pion/kaon FF with bwigm  (formf.f:268)
# Hard-coded parameters (NOT from PKORB — these are formf.f internal).
# ---------------------------------------------------------------------------
def fpikm(W, XM1, XM2):
    S    = W*W
    ROM  = 0.773; ROG  = 0.145
    ROM1 = 1.370; ROG1 = 0.510; BETA1 = -0.145
    return (bwigm(S, ROM, ROG, XM1, XM2) + BETA1*bwigm(S, ROM1, ROG1, XM1, XM2)) / (1. + BETA1)

def fpikmd(W, XM1, XM2):
    S     = W*W
    ROM  = 0.773; ROG  = 0.145
    ROM1 = 1.500; ROG1 = 0.220
    ROM2 = 1.750; ROG2 = 0.120
    BETA  = 6.5; DELTA = -26.0
    return (DELTA*bwigm(S,ROM,ROG,XM1,XM2)
            + BETA*bwigm(S,ROM1,ROG1,XM1,XM2)
            +      bwigm(S,ROM2,ROG2,XM1,XM2)) / (1. + BETA + DELTA)

def fpirk(W):
    return abs(fpikm(W, AMK, AMKZ))**2

# ---------------------------------------------------------------------------
# GFUN — a1 width shape  (tauola.f:3483)
# ---------------------------------------------------------------------------
def gfun(QKWA):
    if QKWA < (AMRO + AMPI)**2:
        x = QKWA - 9.*AMPIZ**2
        return 4.1 * x**3 * (1. - 3.3*x + 5.8*x**2)
    else:
        return QKWA*(1.623 + 10.38/QKWA - 9.32/QKWA**2 + 0.65/QKWA**3)

# ---------------------------------------------------------------------------
# WGA1C, WGA1N, WGA1  (f3pi.f:316-406)
# ---------------------------------------------------------------------------
def wga1c(S):
    STH = 0.1753
    Q0=5.80900; Q1=-3.00980; Q2=4.57920
    P0=-13.91400; P1=27.67900; P2=-13.39300; P3=3.19240; P4=-0.10487
    if S < STH:
        return 0.
    elif S < 0.823:
        d = S - STH
        return Q0*d**3*(1. + Q1*d + Q2*d**2)
    else:
        return P0 + P1*S + P2*S**2 + P3*S**3 + P4*S**4

def wga1n(S):
    STH = 0.1676
    Q0=6.28450; Q1=-2.95950; Q2=4.33550
    P0=-15.41100; P1=32.08800; P2=-17.66600; P3=4.93550; P4=-0.37498
    if S < STH:
        return 0.
    elif S < 0.823:
        d = S - STH
        return Q0*d**3*(1. + Q1*d + Q2*d**2)
    else:
        return P0 + P1*S + P2*S**2 + P3*S**3 + P4*S**4

def wga1(QQ):
    MKST = 0.894; MK = 0.496
    MK1SQ = (MKST+MK)**2; MK2SQ = (MKST-MK)**2
    C3PI  = 0.2384**2; CKST = 4.7621**2*C3PI
    S = float(QQ)
    WG3PIC = wga1c(S); WG3PIN = wga1n(S)
    GKST = 0.
    if S > MK1SQ:
        GKST = math.sqrt((S-MK1SQ)*(S-MK2SQ)) / (2.*S)
    return C3PI*(WG3PIC + WG3PIN) + CKST*GKST

# ---------------------------------------------------------------------------
# FA1A1P — a1+a1prime form factor  (f3pi.f:278)
# ---------------------------------------------------------------------------
_FA1A1P_GG1 = None
_FA1A1P_GG2 = None
_FA1A1P_XM1SQ = None
_FA1A1P_XM2SQ = None
_FA1A1P_BET = None

def _init_fa1a1p():
    global _FA1A1P_GG1, _FA1A1P_GG2, _FA1A1P_XM1SQ, _FA1A1P_XM2SQ, _FA1A1P_BET
    XM1 = AMA1;  XG1 = GAMA1
    XM2 = _P1_17; XG2 = _P2_17; BET = _P3_17
    GG1 = XM1*XG1 / (1.3281*0.806)
    GG2 = XM2*XG2 / (1.3281*0.806)
    _FA1A1P_GG1   = GG1
    _FA1A1P_GG2   = GG2
    _FA1A1P_XM1SQ = XM1*XM1
    _FA1A1P_XM2SQ = XM2*XM2
    _FA1A1P_BET   = complex(BET, 0.)

_init_fa1a1p()

def fa1a1p(XMSQ):
    GF  = wga1(XMSQ)
    FG1 = _FA1A1P_GG1 * GF
    FG2 = _FA1A1P_GG2 * GF
    F1  = complex(-_FA1A1P_XM1SQ, 0.) / complex(XMSQ - _FA1A1P_XM1SQ,  FG1)
    F2  = complex(-_FA1A1P_XM2SQ, 0.) / complex(XMSQ - _FA1A1P_XM2SQ,  FG2)
    return F1 + _FA1A1P_BET*F2

# ---------------------------------------------------------------------------
# F3PI — CLEO a1 form factor  (f3pi.f:4)
# idk=1 → pi-2pi0,  idk=2 → 3pi charged
# ---------------------------------------------------------------------------
_F3PI_BT = None

def _init_f3pi():
    global _F3PI_BT
    _F3PI_BT = [
        complex(1., 0.),
        complex(0.12, 0.) * cmath.exp(complex(0.,  0.99*PI)),
        complex(0.37, 0.) * cmath.exp(complex(0., -0.15*PI)),
        complex(0.87, 0.) * cmath.exp(complex(0.,  0.53*PI)),
        complex(0.71, 0.) * cmath.exp(complex(0.,  0.56*PI)),
        complex(2.10, 0.) * cmath.exp(complex(0.,  0.23*PI)),
        complex(0.77, 0.) * cmath.exp(complex(0., -0.54*PI)),
    ]

_init_f3pi()

def f3pi(IFORM, QQ, SA, SB, idk=2):
    """
    IFORM=1,2 → FORM1/FORM2;  IFORM=3 → FORM3.
    idk=1: pi-2pi0,  idk=2: 3pi
    """
    BT = _F3PI_BT
    MRO=0.7743; GRO=0.1491; MRP=1.370; GRP=0.386
    MF2=1.275;  GF2=0.185;  MF0=1.186; GF0=0.350
    MSG=0.860;  GSG=0.880
    MPIZ=AMPIZ; MPIC=AMPI

    if idk == 1:
        M1=MPIZ; M2=MPIZ; M3=MPIC
    else:
        M1=MPIC; M2=MPIC; M3=MPIC
    M1SQ=M1*M1; M2SQ=M2*M2; M3SQ=M3*M3

    result = complex(0., 0.)

    if IFORM in (1, 2):
        S1 = SA; S2 = SB
        S3 = QQ - SA - SB + M1SQ + M2SQ + M3SQ
        if S3 <= 0. or S2 <= 0.:
            pass
        elif idk == 1:
            F134 = -(1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F150 =  (1./18.)*(QQ-M3SQ+S3)*(2.*M1SQ+2.*M2SQ-S3)/S3
            F167 =  (2./3.)
            FRO1 = bwigml(S1,MRO,GRO,M2,M3,1)
            FRP1 = bwigml(S1,MRP,GRP,M2,M3,1)
            FRO2 = bwigml(S2,MRO,GRO,M3,M1,1)
            FRP2 = bwigml(S2,MRP,GRP,M3,M1,1)
            FF23 = bwigml(S3,MF2,GF2,M1,M2,2)
            FSG3 = bwigml(S3,MSG,GSG,M1,M2,0)
            FF03 = bwigml(S3,MF0,GF0,M1,M2,0)
            result = (BT[0]*FRO1 + BT[1]*FRP1
                    + BT[2]*complex(F134,0.)*FRO2 + BT[3]*complex(F134,0.)*FRP2
                    + BT[4]*complex(F150,0.)*FF23
                    + BT[5]*complex(F167,0.)*FSG3 + BT[6]*complex(F167,0.)*FF03)
        else:  # idk == 2
            F134 = -(1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F15A = -(1./2.)*((S2-M2SQ)-(S3-M3SQ))
            F15B = -(1./18.)*(QQ-M2SQ+S2)*(2.*M1SQ+2.*M3SQ-S2)/S2
            F167 = -(2./3.)
            FRO1 = bwigml(S1,MRO,GRO,M2,M3,1)
            FRP1 = bwigml(S1,MRP,GRP,M2,M3,1)
            FRO2 = bwigml(S2,MRO,GRO,M3,M1,1)
            FRP2 = bwigml(S2,MRP,GRP,M3,M1,1)
            FF21 = bwigml(S1,MF2,GF2,M2,M3,2)
            FF22 = bwigml(S2,MF2,GF2,M3,M1,2)
            FSG2 = bwigml(S2,MSG,GSG,M3,M1,0)
            FF02 = bwigml(S2,MF0,GF0,M3,M1,0)
            result = (BT[0]*FRO1 + BT[1]*FRP1
                    + BT[2]*complex(F134,0.)*FRO2 + BT[3]*complex(F134,0.)*FRP2
                    - BT[4]*complex(F15A,0.)*FF21 - BT[4]*complex(F15B,0.)*FF22
                    - BT[5]*complex(F167,0.)*FSG2 - BT[6]*complex(F167,0.)*FF02)

    elif IFORM == 3:
        S3 = SA; S1 = SB
        S2 = QQ - SA - SB + M1SQ + M2SQ + M3SQ
        if S1 <= 0. or S2 <= 0.:
            pass
        elif idk == 1:
            F34A = (1./3.)*((S2-M2SQ)-(S3-M3SQ))
            F34B = (1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F35  =-(1./2.)*((S1-M1SQ)-(S2-M2SQ))
            FRO1 = bwigml(S1,MRO,GRO,M2,M3,1)
            FRP1 = bwigml(S1,MRP,GRP,M2,M3,1)
            FRO2 = bwigml(S2,MRO,GRO,M3,M1,1)
            FRP2 = bwigml(S2,MRP,GRP,M3,M1,1)
            FF23 = bwigml(S3,MF2,GF2,M1,M2,2)
            result = (BT[2]*(complex(F34A,0.)*FRO1 + complex(F34B,0.)*FRO2)
                    + BT[3]*(complex(F34A,0.)*FRP1 + complex(F34B,0.)*FRP2)
                    + BT[4]*complex(F35,0.)*FF23)
        else:  # idk == 2
            F34A = (1./3.)*((S2-M2SQ)-(S3-M3SQ))
            F34B = (1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F35A = -(1./18.)*(QQ-M1SQ+S1)*(2.*M2SQ+2.*M3SQ-S1)/S1
            F35B =  (1./18.)*(QQ-M2SQ+S2)*(2.*M3SQ+2.*M1SQ-S2)/S2
            F36A = -(2./3.)
            F36B =  (2./3.)
            FRO1 = bwigml(S1,MRO,GRO,M2,M3,1)
            FRP1 = bwigml(S1,MRP,GRP,M2,M3,1)
            FRO2 = bwigml(S2,MRO,GRO,M3,M1,1)
            FRP2 = bwigml(S2,MRP,GRP,M3,M1,1)
            FF21 = bwigml(S1,MF2,GF2,M2,M3,2)
            FF22 = bwigml(S2,MF2,GF2,M3,M1,2)
            FSG1 = bwigml(S1,MSG,GSG,M2,M3,0)
            FSG2 = bwigml(S2,MSG,GSG,M3,M1,0)
            FF01 = bwigml(S1,MF0,GF0,M2,M3,0)
            FF02 = bwigml(S2,MF0,GF0,M3,M1,0)
            result = (BT[2]*(complex(F34A,0.)*FRO1 + complex(F34B,0.)*FRO2)
                    + BT[3]*(complex(F34A,0.)*FRP1 + complex(F34B,0.)*FRP2)
                    - BT[4]*(complex(F35A,0.)*FF21 + complex(F35B,0.)*FF22)
                    - BT[5]*(complex(F36A,0.)*FSG1 + complex(F36B,0.)*FSG2)
                    - BT[6]*(complex(F36A,0.)*FF01 + complex(F36B,0.)*FF02))

    return result * fa1a1p(QQ)

# ---------------------------------------------------------------------------
# FK1AB — K1(1270)/K1(1402) form factor  (formf.f:41)
# INDX=1..4 → PKORB(3,81..88) pairs
# ---------------------------------------------------------------------------
_FK1AB_GG1 = None; _FK1AB_GG2 = None
_FK1AB_XM1SQ = None; _FK1AB_XM2SQ = None

def _init_fk1ab():
    global _FK1AB_GG1, _FK1AB_GG2, _FK1AB_XM1SQ, _FK1AB_XM2SQ
    XM1 = _P1_19; XG1 = _P2_19
    XM2 = _P1_20; XG2 = _P2_20
    GF1 = gfun(XM1*XM1); GF2 = gfun(XM2*XM2)
    _FK1AB_GG1   = XM1*XG1 / GF1
    _FK1AB_GG2   = XM2*XG2 / GF2
    _FK1AB_XM1SQ = XM1*XM1
    _FK1AB_XM2SQ = XM2*XM2

_init_fk1ab()

def fk1ab(XMSQ, INDX):
    """INDX=1..4 matching PKORB(3,81..88) pairs."""
    AMP_MAP = {1: (0,1), 2: (2,3), 3: (4,5), 4: (6,7)}
    ia, ib = AMP_MAP[INDX]
    AMPA = complex(_FK1AB_AMPS[ia], 0.)
    AMPB = complex(_FK1AB_AMPS[ib], 0.)
    GF  = gfun(XMSQ)
    FG1 = _FK1AB_GG1 * GF
    FG2 = _FK1AB_GG2 * GF
    F1  = complex(-_FK1AB_XM1SQ, 0.) / complex(XMSQ - _FK1AB_XM1SQ,  FG1)
    F2  = complex(-_FK1AB_XM2SQ, 0.) / complex(XMSQ - _FK1AB_XM2SQ,  FG2)
    return AMPA*F1 + AMPB*F2

# ---------------------------------------------------------------------------
# FORM1..5 (IVER=0, CLEO mode)  (formf.f:89..)
# ---------------------------------------------------------------------------
def form1(MNUM, QQ, S1, SDWA, idk=2):
    if MNUM == 0:
        return f3pi(1, QQ, S1, SDWA, idk)
    elif MNUM in (1, 2, 3):
        return fa1a1p(QQ) * bwigm(S1, AMKST, GAMKST, AMPI, AMK)
    elif MNUM == 4:
        return fk1ab(QQ, 3) * bwigm(S1, AMKST, GAMKST, AMPI, AMK)
    elif MNUM == 5:
        return fk1ab(QQ, 4) * fpikm(math.sqrt(S1), AMPI, AMPI)
    elif MNUM == 6:
        return fk1ab(QQ, 1) * bwigm(S1, AMKST, GAMKST, AMK, AMPI)
    else:
        return complex(0., 0.)

def form2(MNUM, QQ, S1, SDWA, idk=2):
    if MNUM == 0:
        return f3pi(2, QQ, S1, SDWA, idk)
    elif MNUM in (1, 2, 3):
        return fa1a1p(QQ) * fpikm(math.sqrt(S1), AMK, AMK)
    elif MNUM == 4:
        return fk1ab(QQ, 3) * bwigm(S1, AMKST, GAMKST, AMPI, AMK)
    elif MNUM == 5:
        return fk1ab(QQ, 1) * bwigm(S1, AMKST, GAMKST, AMPI, AMK)
    elif MNUM == 6:
        return fk1ab(QQ, 2) * fpikm(math.sqrt(S1), AMPI, AMPI)
    else:
        return complex(0., 0.)

def form3(MNUM, QQ, S1, SDWA, idk=2):
    if MNUM == 0:
        return f3pi(3, QQ, S1, SDWA, idk)
    elif MNUM == 3:
        return fa1a1p(QQ) * bwigm(S1, AMKST, GAMKST, AMPIZ, AMK)
    elif MNUM == 6:
        return fk1ab(QQ, 3) * bwigm(S1, AMKST, GAMKST, AMK, AMPI)
    else:
        return complex(0., 0.)

def form4(MNUM, QQ, S1, S2, S3):
    # Always 0 for IVER=0
    return complex(0., 0.)

def form5(MNUM, QQ, S1, S2):
    if MNUM == 0:
        return complex(0., 0.)
    elif MNUM in (1, 2):
        ELPHA = -0.2
        return (fpikmd(math.sqrt(QQ), AMPI, AMPI) / (1.+ELPHA)
                * (fpikm(math.sqrt(S2), AMPI, AMPI)
                   + ELPHA*bwigm(S1, AMKST, GAMKST, AMPI, AMK)))
    elif MNUM == 3:
        return complex(0., 0.)
    elif MNUM == 4:
        return complex(0., 0.)
    elif MNUM == 5:
        ELPHA = -0.2
        return (bwigm(QQ, AMKST, GAMKST, AMPI, AMK) / (1.+ELPHA)
                * (fpikm(math.sqrt(S1), AMPI, AMPI)
                   + ELPHA*bwigm(S2, AMKST, GAMKST, AMPI, AMK)))
    elif MNUM == 6:
        ELPHA = -0.2
        return (bwigm(QQ, AMKST, GAMKST, AMPI, AMKZ) / (1.+ELPHA)
                * (fpikm(math.sqrt(S2), AMPI, AMPI)
                   + ELPHA*bwigm(S1, AMKST, GAMKST, AMPI, AMK)))
    elif MNUM == 7:
        return fpikmd(math.sqrt(QQ), AMPI, AMPI) * fpikm(math.sqrt(S1), AMPI, AMPI)
    else:
        return complex(0., 0.)

# ---------------------------------------------------------------------------
# CLVEC, CLAXI, CLNUT, PROD5
# All 4-vectors 0-indexed: [px,py,pz,E]
# idff = sign: +1 for tau+ (pdgid=-15), -1 for tau- (pdgid=15)
#   BUT in original TAUOLA, IDFF=-1 for tau- and +1 for tau+
# KTOM=1 always (set in calculateHH)
# ---------------------------------------------------------------------------

def clvec(hj, pn):
    """Compute vector part PIV from hadronic current hj and neutrino pn."""
    # HN = HJ[3]*PN[3] - HJ[2]*PN[2]  (z and E components only, since PN along z)
    HN   = hj[3]*complex(pn[3]) - hj[2]*complex(pn[2])
    HH_r = (hj[3]*hj[3].conjugate() - hj[2]*hj[2].conjugate()
            - hj[1]*hj[1].conjugate() - hj[0]*hj[0].conjugate()).real
    return [4.*(HN*hj[i].conjugate()).real - 2.*HH_r*pn[i] for i in range(4)]

def claxi(hj, pn, idff):
    """Compute axial-vector part PIA. KTOM=1."""
    sign = idff / abs(idff) if idff != 0 else 1.
    hjc = [hj[i].conjugate() for i in range(4)]
    def det2(i, j):
        return (hjc[i]*hj[j] - hjc[j]*hj[i]).imag
    pia = [0.]*4
    pia[0] = -2.*pn[2]*det2(1,3) + 2.*pn[3]*det2(1,2)
    pia[1] = -2.*pn[3]*det2(0,2) + 2.*pn[2]*det2(0,3)
    pia[2] =  2.*pn[3]*det2(0,1)
    pia[3] =  2.*pn[2]*det2(0,1)
    return [x*sign for x in pia]

def clnut(hj, idff):
    """Compute neutrino-mass contribution (AMNUTA=0 so B≈0 but HV is still computed)."""
    p_unit = [0., 0., 0., 1.]
    hv = claxi(hj, p_unit, idff)
    B  = (hj[3].real*hj[3].imag - hj[2].real*hj[2].imag
          - hj[1].real*hj[1].imag - hj[0].real*hj[0].imag)
    return B, hv

def prod5(p1, p2, p3, idff):
    """External product of 3 four-vectors (Levi-Civita), sign from idff/KTOM=1."""
    sign = idff / abs(idff) if idff != 0 else 1.
    def det2(i, j): return p1[i]*p2[j] - p2[i]*p1[j]
    pia = [0.]*4
    pia[0] = (-p3[2]*det2(1,3) + p3[3]*det2(1,2) + p3[1]*det2(2,3))
    pia[1] = (-p3[3]*det2(0,2) + p3[2]*det2(0,3) - p3[0]*det2(2,3))
    pia[2] = ( p3[3]*det2(0,1) - p3[1]*det2(0,3) + p3[0]*det2(1,3))
    pia[3] = ( p3[2]*det2(0,1) - p3[1]*det2(0,2) + p3[0]*det2(1,2))
    return [x*sign for x in pia]

# ---------------------------------------------------------------------------
# 2-body hadronic currents
# ---------------------------------------------------------------------------

def _transverse_qq(pc, pn_pion):
    """Compute transverse part of (pc - pn_pion) w.r.t. (pc + pn_pion)."""
    pks = [pc[i]+pn_pion[i] for i in range(4)]
    qq  = [pc[i]-pn_pion[i] for i in range(4)]
    pksd  = pks[3]**2 - pks[2]**2 - pks[1]**2 - pks[0]**2
    qqpks = pks[3]*qq[3] - pks[2]*qq[2] - pks[1]*qq[1] - pks[0]*qq[0]
    return [qq[i] - pks[i]*qqpks/pksd for i in range(4)], pksd

def curr_pipi0(pc, pn_pion):
    """IVER=0: HADCUR = QQ * sqrt(fpirho(sqrt(PKSD)))"""
    qq, pksd = _transverse_qq(pc, pn_pion)
    ff = math.sqrt(fpirho(math.sqrt(pksd)))
    return [complex(qq[i]*ff) for i in range(4)]

def curr_pik0(pc, pn_pion):
    """HADCUR = QQ * bwigs(PKSD, AMKST, GAMKST)"""
    qq, pksd = _transverse_qq(pc, pn_pion)
    ff = bwigs(pksd, AMKST, GAMKST)
    return [complex(qq[i])*ff for i in range(4)]

def curr_kpi0(pc, pn_pion):
    qq, pksd = _transverse_qq(pc, pn_pion)
    ff = bwigs(pksd, AMKST, GAMKST)
    return [complex(qq[i])*ff for i in range(4)]

def curr_kk0(pc, pn_pion):
    qq, pksd = _transverse_qq(pc, pn_pion)
    ff = math.sqrt(fpirk(math.sqrt(pksd)))
    return [complex(qq[i]*ff) for i in range(4)]

# ---------------------------------------------------------------------------
# DAM2PI
# ---------------------------------------------------------------------------

def dam2pi(MNUM, PT, PN, PIM1, PIM2, idff):
    """
    Returns (AMPLIT, HV[4]) where HV[3]=0.
    MNUM: 0=pi-pi0, 1=pi-K0, 2=K-pi0, 3=K-K0
    """
    if   MNUM == 0: hadcur = curr_pipi0(PIM1, PIM2)
    elif MNUM == 1: hadcur = curr_pik0 (PIM1, PIM2)
    elif MNUM == 2: hadcur = curr_kpi0 (PIM1, PIM2)
    elif MNUM == 3: hadcur = curr_kk0  (PIM1, PIM2)
    else: raise ValueError(f"DAM2PI: bad MNUM={MNUM}")

    pivec = clvec(hadcur, PN)
    piaks = claxi(hadcur, PN, idff)
    brakm, hvm = clnut(hadcur, idff)

    BRAK = ((GV**2+GA**2)*PT[3]*pivec[3]
            + 2.*GV*GA*PT[3]*piaks[3]
            + 2.*(GV**2-GA**2)*AMNUTA*AMTAU*brakm)
    if MNUM in (0, 3):
        AMPLIT = (CCABIB*GFERMI)**2 * BRAK
    else:
        AMPLIT = (SCABIB*GFERMI)**2 * BRAK

    hv = [0.]*4
    for i in range(3):
        hv_i = (-(AMTAU*((GV**2+GA**2)*piaks[i] + 2.*GV*GA*pivec[i]))
                + (GV**2-GA**2)*AMNUTA*AMTAU*hvm[i])
        hv[i] = -hv_i / BRAK
    hv[3] = 0.
    return AMPLIT, hv

# ---------------------------------------------------------------------------
# DILOGT (tauola.f:5702) — Chebyshev expansion
# ---------------------------------------------------------------------------
def dilogt(X):
    """Dilogarithm — direct translation of CERN C304 / tauola.f:5702."""
    Z = -1.64493406684822
    if X < -1.0:
        Z = 3.2898681336964; T = 1./X; S = -0.5
        Z -= 0.5*math.log(abs(X))**2
    elif X <= 0.5:
        T = X; S = 0.5; Z = 0.
    elif X == 1.0:
        return 1.64493406684822
    elif X <= 2.0:
        T = 1.-X; S = -0.5
        Z = 1.64493406684822 - math.log(X)*math.log(abs(T))
    else:
        Z = 3.2898681336964; T = 1./X; S = -0.5
        Z -= 0.5*math.log(abs(X))**2

    Y = 2.66666666666666*T + 0.66666666666666
    B = 1.e-17
    A = Y*B + 4.e-17
    for ca, cb in [
        (1.1e-16,  3.7e-16),(1.21e-15,  3.98e-15),
        (1.312e-14,4.342e-14),(1.4437e-13,4.8274e-13),
        (1.62421e-12,5.50291e-12),(1.879117e-11,6.474338e-11),
        (2.2536705e-10,7.9387055e-10),(2.83575385e-9,1.029904264e-8),
        (3.816329463e-8,1.4496300557e-7),(5.6817822718e-7,2.32002196094e-6),
        (1.001627496164e-5,4.686361959447e-5),(2.4879322924228e-4,1.66073032927855e-3),
    ]:
        B, A = Y*A - B + ca, Y*B - A + cb
    A = Y*A - B + 1.93506430086996
    return S*T*(A - B) + Z

# ---------------------------------------------------------------------------
# THB — Born+QED matrix element for leptonic decay  (tauola.f:1299)
# ---------------------------------------------------------------------------
def _thb(ITDKRC, QP, XN, XA, AK0):
    """
    Returns (AMPLIT, HV[3]) for tau → lep nu nubar.
    HV[3] (4th component) is set to 1.0 by DAMPRY.
    """
    TMASS  = AMTAU
    GF     = GFERMI
    ALPHAI = ALFINV
    TMASS2 = TMASS**2

    # Scalar products
    def dot(a, b):
        return a[3]*b[3] - a[2]*b[2] - a[1]*b[1] - a[0]*b[0]

    QPXN = dot(QP, XN)
    QPXA = dot(QP, XA)
    XNXA = dot(XN, XA)
    TXN  = TMASS * XN[3]
    TXA  = TMASS * XA[3]
    TQP  = TMASS * QP[3]

    BRAK = ((GV+GA)**2*TQP*XNXA + (GV-GA)**2*TXA*QPXN
            - (GV**2-GA**2)*TMASS*AMNUTA*QPXA)
    BORN = 32.*(GF**2/2.)*BRAK

    AM3 = 0.
    AM3POL = [0., 0., 0.]

    if ITDKRC != 0:
        U0 = QP[3]/TMASS
        U3 = math.sqrt(QP[0]**2 + QP[1]**2 + QP[2]**2) / TMASS
        W3 = U3
        W0 = (XN[3] + XA[3]) / TMASS
        UP = U0+U3; UM = U0-U3
        WP = W0+W3; WM = W0-W3
        YU = math.log(UP/UM) / 2.
        YW = math.log(WP/WM) / 2.
        EPS2 = U0**2 - U3**2
        EPS  = math.sqrt(EPS2)
        Y    = W0**2 - W3**2
        AL   = AK0/TMASS

        F0 = (2.*U0/U3*(dilogt(1.-(UM*WM/(UP*WP))) - dilogt(1.-WM/WP)
              + dilogt(1.-UM/UP) - 2.*YU + 2.*math.log(UP)*(YW+YU))
              + 1./Y*(2.*U3*YU + (1.-EPS2-2.*Y)*math.log(EPS))
              + 2. - 4.*(U0/U3*YU - 1.)*math.log(2.*AL))
        FP = YU/(2.*U3)*(1. + (1.-EPS2)/Y) + math.log(EPS)/Y
        FM = YU/(2.*U3)*(1. - (1.-EPS2)/Y) - math.log(EPS)/Y
        F3 = EPS2*(FP+FM)/2.

        CONST3 = 1./(2.*ALPHAI*PI)*64.*GF**2
        XM3 = -(F0*QPXN*TXA + FP*EPS2*TXN*TXA + FM*QPXN*QPXA + F3*TMASS2*XNXA)
        AM3 = XM3 * CONST3

        # Polarized QED corrections
        for idx in range(3):
            R = [0.,0.,0., TMASS]
            R[idx] = TMASS
            RXA_i = R[3]*XA[3] - R[0]*XA[0] - R[1]*XA[1] - R[2]*XA[2]
            RXN_i = R[3]*XN[3] - R[0]*XN[0] - R[1]*XN[1] - R[2]*XN[2]
            RQP_i = R[3]*QP[3] - R[0]*QP[0] - R[1]*QP[1] - R[2]*QP[2]
            XM3P  = -(F0*QPXN*RXA_i + FP*EPS2*TXN*RXA_i
                      + FM*QPXN*(QPXA + (RXA_i*TQP - TXA*RQP_i)/TMASS2)
                      + F3*(TMASS2*XNXA + TXN*RXA_i - RXN_i*TXA))
            AM3POL[idx] = XM3P * CONST3

    BORNPL = [0.]*3
    for idx in range(3):
        BORNPL[idx] = BORN + (
            (GV+GA)**2*TMASS*XNXA*QP[idx]
            - (GV-GA)**2*TMASS*QPXN*XA[idx]
            + (GV**2-GA**2)*AMNUTA*TXA*QP[idx]
            - (GV**2-GA**2)*AMNUTA*TQP*XA[idx]
        ) * 32.*(GF**2/2.)

    THB_val = BORN + AM3
    if THB_val / BORN < 0.1:
        THB_val = 0.

    hv = [0.]*3
    for i in range(3):
        hv[i] = (BORNPL[i] + AM3POL[i]) / THB_val - 1. if THB_val != 0. else 0.

    return THB_val, hv

# ---------------------------------------------------------------------------
# SQM2 — real photon matrix element  (tauola.f:1221)
# ---------------------------------------------------------------------------
def _sqm2(ITDKRC, QP, XN, XA, XK, AK0):
    TMASS  = AMTAU
    GF     = GFERMI
    ALPHAI = ALFINV
    TMASS2 = TMASS**2
    EMASS2 = QP[3]**2 - QP[2]**2 - QP[1]**2 - QP[0]**2

    def dot(a, b):
        return a[3]*b[3] - a[2]*b[2] - a[1]*b[1] - a[0]*b[0]

    QPXN = dot(QP, XN); QPXA = dot(QP, XA); QPXK = dot(QP, XK)
    XNXK = dot(XN, XK); XAXK = dot(XA, XK)
    TXN  = TMASS*XN[3]; TXA  = TMASS*XA[3]
    TQP  = TMASS*QP[3]; TXK  = TMASS*XK[3]

    X = XNXK/QPXN; Z = TXK/TQP
    A = 1.+X; B = 1.+X*(1.+Z)/2.+Z/2.
    S1 = (QPXN*TXA*(-EMASS2/QPXK**2*A + 2.*TQP/(QPXK*TXK)*B - TMASS2/TXK**2)
          + QPXN/TXK**2*(TMASS2*XAXK - TXA*TXK + XAXK*TXK)
          - TXA*TXN/TXK - QPXN/(QPXK*TXK)*(TQP*XAXK - TXK*QPXA))

    CONST4 = 256.*PI/ALPHAI*GF**2 if ITDKRC != 0 else 0.
    SQM2_val = S1*CONST4

    hv = [0.]*3
    if S1 != 0.:
        for idx in range(3):
            R = [0.,0.,0., TMASS]
            R[idx] = TMASS
            RXA_i = R[3]*XA[3] - R[0]*XA[0] - R[1]*XA[1] - R[2]*XA[2]
            RXK_i = R[3]*XK[3] - R[0]*XK[0] - R[1]*XK[1] - R[2]*XK[2]
            RQP_i = R[3]*QP[3] - R[0]*QP[0] - R[1]*QP[1] - R[2]*QP[2]
            S0_i  = (QPXN*RXA_i*(-EMASS2/QPXK**2*A + 2.*TQP/(QPXK*TXK)*B - TMASS2/TXK**2)
                     + QPXN/TXK**2*(TMASS2*XAXK - TXA*RXK_i + XAXK*RXK_i)
                     - RXA_i*TXN/TXK - QPXN/(QPXK*TXK)*(RQP_i*XAXK - RXK_i*QPXA))
            hv[idx] = S0_i/S1 - 1.
    return SQM2_val, hv

# ---------------------------------------------------------------------------
# DAMPRY — leptonic decay  (tauola.f:1196)
# ---------------------------------------------------------------------------
def dampry(ITDKRC, XK0DEC, XK, XA, QP, XN):
    """
    Returns (AMPLIT, HV[4])  with HV[3]=1.0.
    """
    AK0 = XK0DEC * AMTAU
    if XK[3] < 0.1*AK0:
        amplit, hv3 = _thb(ITDKRC, QP, XN, XA, AK0)
    else:
        amplit, hv3 = _sqm2(ITDKRC, QP, XN, XA, XK, AK0)
    hv = list(hv3) + [1.0]
    return amplit, hv

# ---------------------------------------------------------------------------
# DAMPPK — 3-body hadronic  (tauola.f:3772)
# idk: 1=pi-2pi0 (MNUM=0 only), 2=3pi (MNUM=0 only)
# ---------------------------------------------------------------------------
def damppk(MNUM, PT, PN, PIM1, PIM2, PIM3, idff, idk=2):
    """
    Returns (AMPLIT, HV[4])  with HV[3]=0.
    PIM1=first hadron, PIM2=second, PIM3=third (same order as Fortran DAMPPK).
    """
    FPI  = _FPIc
    FNRM = _fnorm(MNUM)
    UROJ = complex(0., 1.)

    PAA = [PIM1[i]+PIM2[i]+PIM3[i] for i in range(4)]
    XMAA2 = max(_mass2(PAA), 0.)
    XMAA  = math.sqrt(XMAA2)

    def inv_mass2(a, b):
        return max(_mass2([a[i]+b[i] for i in range(4)]), 0.)

    XMRO1_2 = inv_mass2(PIM3, PIM2)
    XMRO2_2 = inv_mass2(PIM3, PIM1)
    XMRO3_2 = inv_mass2(PIM1, PIM2)

    PROD1 = _dot4(PAA, [PIM2[i]-PIM3[i] for i in range(4)])
    PROD2 = _dot4(PAA, [PIM3[i]-PIM1[i] for i in range(4)])
    PROD3 = _dot4(PAA, [PIM1[i]-PIM2[i] for i in range(4)])

    VEC1 = [PIM2[i]-PIM3[i] - PAA[i]*PROD1/XMAA2 for i in range(4)]
    VEC2 = [PIM3[i]-PIM1[i] - PAA[i]*PROD2/XMAA2 for i in range(4)]
    VEC3 = [PIM1[i]-PIM2[i] - PAA[i]*PROD3/XMAA2 for i in range(4)]
    VEC4 = list(PAA)
    VEC5 = prod5(PIM1, PIM2, PIM3, idff)

    F1 = complex(coef(1,MNUM)) * form1(MNUM, XMAA2, XMRO1_2, XMRO2_2, idk)
    F2 = complex(coef(2,MNUM)) * form2(MNUM, XMAA2, XMRO2_2, XMRO1_2, idk)
    F3 = complex(coef(3,MNUM)) * form3(MNUM, XMAA2, XMRO3_2, XMRO1_2, idk)
    F4 = (-1.*UROJ) * complex(coef(4,MNUM)) * form4(MNUM, XMAA2, XMRO1_2, XMRO2_2, XMRO3_2)
    F5 = (-1.*UROJ)/(4.*PI**2*FPI**2) * complex(coef(5,MNUM)) * form5(MNUM, XMAA2, XMRO1_2, XMRO2_2)

    hadcur = [complex(FNRM) * (complex(VEC1[i])*F1 + complex(VEC2[i])*F2
                               + complex(VEC3[i])*F3 + complex(VEC4[i])*F4
                               + complex(VEC5[i])*F5)
              for i in range(4)]

    pivec = clvec(hadcur, PN)
    piaks = claxi(hadcur, PN, idff)
    brakm, hvm = clnut(hadcur, idff)

    BRAK = ((GV**2+GA**2)*PT[3]*pivec[3]
            + 2.*GV*GA*PT[3]*piaks[3]
            + 2.*(GV**2-GA**2)*AMNUTA*AMTAU*brakm)
    AMPLIT = GFERMI**2 * BRAK / 2.

    hv = [0.]*4
    for i in range(3):
        hv_i = (-(AMTAU*((GV**2+GA**2)*piaks[i] + 2.*GV*GA*pivec[i]))
                + (GV**2-GA**2)*AMNUTA*AMTAU*hvm[i])
        hv[i] = -hv_i / BRAK
    hv[3] = 0.
    return AMPLIT, hv

# ---------------------------------------------------------------------------
# CURR_CLEO — 4pi hadronic current  (curr_cleo.f)
# ---------------------------------------------------------------------------
def _init_curr_cleo():
    """Precompute CURR_CLEO constants."""
    G1   = 12.924; G2 = 1475.98; FPI = 93.3e-3
    G    = G1*G2
    FRO  = 0.266*AMRO**2
    COEF1 = 2.*math.sqrt(3.)/FPI**2
    COEF2 = FRO*G*0.56   # factor 0.56 for omega contribution

    AMRO2  = _P1_21; GAMRO2 = _P2_21
    AMRO3  = _P1_22; GAMRO3 = _P2_22
    AMOM   = _P1_14; GAMOM  = _P2_14

    AMPL = [None]*8
    AMPL[1] = complex(_A31*COEF1, 0.)
    AMPL[2] = complex(_A32*COEF1, 0.) * cmath.exp(complex(0., _PH42))
    AMPL[3] = complex(_A33*COEF1, 0.) * cmath.exp(complex(0., _PH43))
    AMPL[4] = complex(_A34*COEF1, 0.) * cmath.exp(complex(0., _PH44))
    AMPL[5] = complex(_A35*COEF2, 0.) * cmath.exp(complex(0., _PH45))
    AMPL[6] = complex(_A36*COEF1, 0.)
    AMPL[7] = complex(_A37*COEF1, 0.)

    ALF0 = complex(_ALF0v, 0.)
    ALF1 = complex(_ALF1v*AMRO**2,  0.)
    ALF2 = complex(_ALF2v*AMRO2**2, 0.)
    ALF3 = complex(_ALF3v*AMRO3**2, 0.)
    LAM0 = complex(_LAM0v, 0.)
    LAM1 = complex(_LAM1v*AMRO**2,  0.)
    LAM2 = complex(_LAM2v*AMRO2**2, 0.)
    LAM3 = complex(_LAM3v*AMRO3**2, 0.)
    BET1 = complex(_BET1v*AMRO**2,  0.)
    BET2 = complex(_BET2v*AMRO2**2, 0.)
    BET3 = complex(_BET3v*AMRO3**2, 0.)

    return (AMRO2, GAMRO2, AMRO3, GAMRO3, AMOM, GAMOM,
            AMPL, ALF0, ALF1, ALF2, ALF3,
            LAM0, LAM1, LAM2, LAM3,
            BET1, BET2, BET3)

_CC = _init_curr_cleo()

def curr_cleo(MNUM, PIM1, PIM2, PIM3, PIM4):
    """
    Returns HADCUR[4] (complex).
    MNUM=1: pi-pi-pi0pi+,  MNUM!=1: pi0pi0pi0pi-
    """
    (AMRO2, GAMRO2, AMRO3, GAMRO3, AMOM, GAMOM,
     AMPL, ALF0, ALF1, ALF2, ALF3,
     LAM0, LAM1, LAM2, LAM3, BET1, BET2, BET3) = _CC

    def bwign(A, XM, XG):
        return 1. / complex(A - XM**2, XM*XG)

    PP = [PIM1, PIM2, PIM3, PIM4]   # PP[k-1] = PP(k) in Fortran
    PAA = [sum(PP[k][i] for k in range(4)) for i in range(4)]
    hadcur = [complex(0.)]*4
    AA = [[0.]*4 for _ in range(4)]

    if MNUM == 1:
        QQ = _mass2(PAA)
        FORM4 = LAM0 + LAM1*bwign(QQ,AMRO,GAMRO) + LAM2*bwign(QQ,AMRO2,GAMRO2) + LAM3*bwign(QQ,AMRO3,GAMRO3)

        for K1 in range(1,4):   # K1=1,2,3
            for K2 in range(3,5):  # K2=3,4
                if K2 == K1:
                    continue
                if K2 == 3:
                    AMPR = AMPL[3]; AMPA = AMPIZ
                elif K1 == 3:
                    AMPR = AMPL[4]; AMPA = AMPIZ
                else:
                    AMPR = AMPL[2]; AMPA = AMPI

                SK = _mass2([PP[K1-1][i]+PP[K2-1][i] for i in range(4)])

                for i in range(4):
                    for j in range(4):
                        AA[i][j] = 1. if i==j else 0.

                for L in range(1,5):  # L=1,2,3,4
                    if L != K1 and L != K2:
                        PL = PP[L-1]
                        DENOM = _mass2([PAA[i]-PL[i] for i in range(4)])
                        for i in range(4):
                            for j in range(4):
                                SIG = 1. if j==3 else -1.
                                AA[i][j] -= SIG*(PAA[i]-2.*PL[i])*(PAA[j]-PL[j])/DENOM

                FORM2PI = (BET1*bwigm(SK, AMRO,  GAMRO,  AMPA, AMPI)
                         + BET2*bwigm(SK, AMRO2, GAMRO2, AMPA, AMPI)
                         + BET3*bwigm(SK, AMRO3, GAMRO3, AMPA, AMPI))
                FORM1_val = AMPL[1] + AMPR*FORM2PI

                for i in range(4):
                    s = complex(0.)
                    for j in range(4):
                        s += FORM1_val * FORM4 * AA[i][j] * (PP[K1-1][j] - PP[K2-1][j])
                    hadcur[i] += s

        # Omega current
        if AMPL[5] != complex(0.):
            FORM2_om = AMPL[5]*(ALF0 + ALF1*bwign(QQ,AMRO,GAMRO)
                                + ALF2*bwign(QQ,AMRO2,GAMRO2) + ALF3*bwign(QQ,AMRO3,GAMRO3))
            for KK in range(1,3):  # KK=1,2
                PA = list(PP[KK-1]); PB = list(PP[2-KK])
                QQA=SS23=SS24=SS34=QP1P2=QP1P3=QP1P4=P1P2=P1P3=P1P4 = 0.
                for K in range(4):
                    SIGN = 1. if K==3 else -1.
                    QQA   += SIGN*(PAA[K]-PA[K])**2
                    SS23  += SIGN*(PB[K]+PIM3[K])**2
                    SS24  += SIGN*(PB[K]+PIM4[K])**2
                    SS34  += SIGN*(PIM3[K]+PIM4[K])**2
                    QP1P2 += SIGN*(PAA[K]-PA[K])*PB[K]
                    QP1P3 += SIGN*(PAA[K]-PA[K])*PIM3[K]
                    QP1P4 += SIGN*(PAA[K]-PA[K])*PIM4[K]
                    P1P2  += SIGN*PA[K]*PB[K]
                    P1P3  += SIGN*PA[K]*PIM3[K]
                    P1P4  += SIGN*PA[K]*PIM4[K]
                FORM3_om = bwign(QQA, AMOM, GAMOM)
                for K in range(4):
                    hadcur[K] += FORM2_om*FORM3_om*(
                        PB[K]  *(QP1P3*P1P4 - QP1P4*P1P3)
                       +PIM3[K]*(QP1P4*P1P2 - QP1P2*P1P4)
                       +PIM4[K]*(QP1P2*P1P3 - QP1P3*P1P2))

    else:  # pi0pi0pi0pi- case
        QQ = _mass2(PAA)
        for K in range(1,4):  # K=1,2,3
            SK = _mass2([PP[K-1][i]+PP[3][i] for i in range(4)])

            for i in range(4):
                for j in range(4):
                    AA[i][j] = 1. if i==j else 0.

            for L in range(1,4):  # L=1,2,3
                if L != K:
                    PL = PP[L-1]
                    DENOM = _mass2([PAA[i]-PL[i] for i in range(4)])
                    for i in range(4):
                        for j in range(4):
                            SIG = 1. if j==3 else -1.
                            AA[i][j] -= SIG*(PAA[i]-2.*PL[i])*(PAA[j]-PL[j])/DENOM

            FORM1_val = AMPL[6] + AMPL[7]*fpikm(math.sqrt(SK), AMPI, AMPI)

            for i in range(4):
                s = complex(0.)
                for j in range(4):
                    s += FORM1_val * AA[i][j] * (PP[K-1][j] - PP[3][j])
                hadcur[i] += s

    return hadcur

# ---------------------------------------------------------------------------
# DAM4PI — 4-pi hadronic  (tauola.f:4447)
# ---------------------------------------------------------------------------
def dam4pi(MNUM, PT, PN, PIM1, PIM2, PIM3, PIM4, idff):
    """
    Returns (AMPLIT, HV[4])  with HV[3]=0.
    MNUM=1: pi-pi-pi0pi+,  MNUM=2: pi0pi0pi0pi-
    """
    hadcur = curr_cleo(MNUM, PIM1, PIM2, PIM3, PIM4)

    pivec = clvec(hadcur, PN)
    piaks = claxi(hadcur, PN, idff)
    brakm, hvm = clnut(hadcur, idff)

    BRAK = ((GV**2+GA**2)*PT[3]*pivec[3]
            + 2.*GV*GA*PT[3]*piaks[3]
            + 2.*(GV**2-GA**2)*AMNUTA*AMTAU*brakm)
    AMPLIT = (CCABIB*GFERMI)**2 * BRAK / 2.

    hv = [0.]*4
    for i in range(3):
        hv_i = (-(AMTAU*((GV**2+GA**2)*piaks[i] + 2.*GV*GA*pivec[i]))
                + (GV**2-GA**2)*AMNUTA*AMTAU*hvm[i])
        hv[i] = -hv_i / BRAK
    hv[3] = 0.
    return AMPLIT, hv
