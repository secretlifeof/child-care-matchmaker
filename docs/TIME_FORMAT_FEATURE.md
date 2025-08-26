# Time String Format Feature

## Overview

The TimeSlot model now supports both the original minutes-based format and a new user-friendly time string format using "HH:MM" notation.

## Features

### 1. Time String Format (New)
Use intuitive "HH:MM" format strings:

```json
{
  "day_of_week": 0,
  "start": "07:30",
  "end": "16:30"
}
```

### 2. Minutes Format (Backward Compatible)
Original format using minutes from start of day:

```json
{
  "day_of_week": 0,
  "start_minute": 450,
  "end_minute": 990
}
```

### 3. Mixed Format Support
You can mix both formats in the same request:

```json
{
  "desired_hours": [
    {
      "day_of_week": 0,
      "start": "07:30",
      "end": "16:30"
    },
    {
      "day_of_week": 1,
      "start_minute": 480,
      "end_minute": 960
    }
  ]
}
```

## API Changes

### TimeSlot Model
The `TimeSlot` class has been enhanced with:

- **New fields**: `start` and `end` (optional string fields)
- **Validation**: Time strings must match `HH:MM` format (00:00 to 23:59)
- **Automatic conversion**: Time strings are automatically converted to minutes internally
- **New properties**: `start_time_string` and `end_time_string` for getting "HH:MM" format
- **New class method**: `TimeSlot.from_time_strings(day_of_week, start, end)`

### Validation Rules

1. **Either format required**: Each TimeSlot must have either:
   - `start` and `end` (time strings), OR
   - `start_minute` and `end_minute` (integers)

2. **Time string format**: Must match `HH:MM` pattern:
   - Hours: 00-23
   - Minutes: 00-59
   - Examples: `"07:30"`, `"16:45"`, `"23:59"`

3. **Minutes range**: 
   - `start_minute`: 0-1439 (0 to 23:59)
   - `end_minute`: 0-1440 (0 to 24:00)

## Usage Examples

### Basic Time String Usage

```python
from src.matchmaker.models.base import TimeSlot

# Create with time strings
slot = TimeSlot(day_of_week=0, start="07:30", end="16:30")

# Access converted values
print(slot.start_minute)      # 450
print(slot.end_minute)        # 990
print(slot.start_time_string) # "07:30"
print(slot.end_time_string)   # "16:30"
```

### Using Class Methods

```python
# Create from time strings
slot = TimeSlot.from_time_strings(
    day_of_week=1, 
    start="08:00", 
    end="17:00"
)

# Create from hours (backward compatibility)
slot = TimeSlot.from_hours(
    day_of_week=2, 
    start_hour=8, 
    end_hour=17, 
    start_min=30, 
    end_min=15
)
```

### JSON Request Examples

#### Recommendation Request with Time Strings
```json
{
  "application": {
    "desired_hours": [
      {
        "day_of_week": 0,
        "start": "07:30",
        "end": "16:30"
      },
      {
        "day_of_week": 1,
        "start": "08:00",
        "end": "17:00"
      }
    ]
  },
  "centers": [
    {
      "opening_hours": [
        {
          "day_of_week": 0,
          "start": "06:30",
          "end": "18:00"
        }
      ]
    }
  ]
}
```

## Time Conversion Reference

| Time String | Minutes | Calculation |
|-------------|---------|-------------|
| "00:00"     | 0       | 0*60 + 0    |
| "07:30"     | 450     | 7*60 + 30   |
| "08:00"     | 480     | 8*60 + 0    |
| "12:15"     | 735     | 12*60 + 15  |
| "16:30"     | 990     | 16*60 + 30  |
| "23:59"     | 1439    | 23*60 + 59  |

## Migration Guide

### For Existing Applications
- **No changes required**: Existing minute-based format continues to work
- **Gradual adoption**: You can start using time strings in new requests
- **Mixed usage**: Use both formats in the same application if needed

### For New Applications
- **Recommended**: Use time string format for better readability
- **Validation**: Ensure time strings follow "HH:MM" format
- **Consistency**: Consider using one format consistently across your application

## Error Handling

### Invalid Time Strings
```python
# These will raise ValidationError
TimeSlot(day_of_week=0, start="25:00", end="16:00")  # Invalid hour
TimeSlot(day_of_week=0, start="10:60", end="16:00")  # Invalid minute
TimeSlot(day_of_week=0, start="8:30", end="16:00")   # Missing leading zero
```

### Missing Time Fields
```python
# These will raise ValueError
TimeSlot(day_of_week=0, end="16:00")    # Missing start time
TimeSlot(day_of_week=0, start="08:00")  # Missing end time
TimeSlot(day_of_week=0)                 # Missing both times
```

## Testing

The feature includes comprehensive tests covering:
- Basic time string functionality
- Backward compatibility with minutes format
- Mixed format usage
- Validation and error handling
- Integration with the complete matching system

Run tests with:
```bash
docker-compose --profile test run --rm matchmaker-test python -m pytest tests/test_time_string_format.py -v
```

## Internal Implementation

### Conversion Process
1. When a TimeSlot is created with time strings, they are automatically converted to minutes
2. The original string values are stored in the `start` and `end` fields
3. All internal logic continues to use the minute values
4. Properties provide access to both formats

### Performance Impact
- **Minimal**: Conversion happens once at object creation
- **Memory**: Slight increase due to storing both formats
- **Compatibility**: No impact on existing functionality

## Future Enhancements

Potential future improvements:
- Support for seconds precision ("HH:MM:SS")
- Time zone awareness
- Duration-based specifications
- Custom time formats