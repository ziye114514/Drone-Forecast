"""Current relative support need not calibrate a new whole-object reference."""
import copy


class QualifiedReferenceResponse:
    def __init__(self,response):self.response=response
    def step(self,row):
        if (row.get('periodic_track_continuation') or {}).get('used'):
            row=copy.deepcopy(row)
            row['body']['native_reference_calibration_allowed']=False
            row['body']['native_reference_qualification_reason']='continued_support_is_not_a_new_complete_material_reference'
        return self.response.step(row)
