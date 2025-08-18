# src/serving/schemas.py
from pydantic import BaseModel
from typing import Literal, Optional

class InputData(BaseModel):
    BeneID: str
    ClaimID: str
    ClaimStartDt: str
    ClaimEndDt: str
    Provider: str
    InscClaimAmtReimbursed: int
    AttendingPhysician: str
    OperatingPhysician: Optional[str] = None
    OtherPhysician: Optional[str] = None
    AdmissionDt: Optional[str] = None
    ClmAdmitDiagnosisCode: Optional[str] = None
    DeductibleAmtPaid: float
    DischargeDt: Optional[str] = None
    DiagnosisGroupCode: Optional[str] = None
    ClmDiagnosisCode_1: Optional[str] = None
    ClmDiagnosisCode_2: Optional[str] = None
    ClmDiagnosisCode_3: Optional[str] = None
    ClmDiagnosisCode_4: Optional[str] = None
    ClmDiagnosisCode_5: Optional[str] = None
    ClmDiagnosisCode_6: Optional[str] = None
    ClmDiagnosisCode_7: Optional[str] = None
    ClmDiagnosisCode_8: Optional[str] = None
    ClmDiagnosisCode_9: Optional[str] = None
    ClmDiagnosisCode_10: Optional[str] = None
    ClmProcedureCode_1: Optional[str] = None
    ClmProcedureCode_2: Optional[str] = None
    ClmProcedureCode_3: Optional[str] = None
    ClmProcedureCode_4: Optional[str] = None
    ClmProcedureCode_5: Optional[str] = None
    ClmProcedureCode_6: Optional[str] = None
    Type: str
    DOB: str
    DOD: Optional[str] = None
    Gender: int
    Race: int
    RenalDiseaseIndicator: str
    State: int
    County: int
    NoOfMonths_PartACov: int
    NoOfMonths_PartBCov: int
    ChronicCond_Alzheimer: int
    ChronicCond_Heartfailure: int
    ChronicCond_KidneyDisease: int
    ChronicCond_Cancer: int
    ChronicCond_ObstrPulmonary: int
    ChronicCond_Depression: int
    ChronicCond_Diabetes: int
    ChronicCond_IschemicHeart: int
    ChronicCond_Osteoporasis: int
    ChronicCond_rheumatoidarthritis: int
    ChronicCond_stroke: int
    IPAnnualReimbursementAmt: int
    IPAnnualDeductibleAmt: int
    OPAnnualReimbursementAmt: int
    OPAnnualDeductibleAmt: int
    PotentialFraud: str

class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
