class WeeklyScheduleRequest(BaseModel):
    center_id: UUID
    week_start_date: date
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    constraints: Optional[List[ScheduleConstraint]] = []
    optimization_config: Optional[EnhancedOptimizationConfig] = None


class MonthlyScheduleRequest(BaseModel):
    center_id: UUID
    month_start_date: date
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    constraints: Optional[List[ScheduleConstraint]] = []
    optimization_config: Optional[EnhancedOptimizationConfig] = None


class MultiWeekScheduleRequest(BaseModel):
    center_id: UUID
    start_date: date
    number_of_weeks: int = Field(ge=1, le=52)
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    optimization_config: Optional[EnhancedOptimizationConfig] = None


class OptimizeExistingScheduleRequest(BaseModel):
    center_id: UUID
    current_schedule: List[ScheduledShift]
    target_start_date: date
    target_end_date: Optional[date] = None
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    constraints: Optional[List[ScheduleConstraint]] = []
    optimization_config: Optional[EnhancedOptimizationConfig] = None
    optimization_goals: Optional[List[OptimizationGoal]] = None


class BulkScheduleRequest(BaseModel):
    center_id: UUID
    schedule_start_date: date
    schedule_end_date: Optional[date] = None
    staff: List[Staff]
    groups: List[Group]
    staffing_requirements: List[StaffingRequirement]
    optimization_config: Optional[EnhancedOptimizationConfig] = None