# Matching Algorithms Deep Dive

This document explains the graph-based matching algorithms used in the Parent-Daycare Matchmaker service.

## Overview

The service uses a **bipartite graph** where:
- **Left nodes**: Applications (parents seeking daycare)
- **Right nodes**: Capacity buckets (available slots in centers)
- **Edges**: Weighted connections representing match quality

Three algorithms operate on this graph for different use cases.

## Graph Construction

### Node Types

#### Application Nodes
Each application becomes a single node with metadata:
```python
{
  "id": "app_uuid",
  "type": "application", 
  "bipartite": 0,  # Left side
  "application": Application,  # Full object
  "demand": 1  # Always wants 1 placement
}
```

#### Capacity Bucket Nodes
Centers are decomposed into capacity buckets:
```python
{
  "id": "bucket_uuid",
  "type": "bucket",
  "bipartite": 1,  # Right side
  "center": Center,  # Parent center
  "bucket": CapacityBucket,  # Specific bucket
  "capacity": 10  # Available slots
}
```

### Edge Creation

Edges are created between compatible applications and buckets:

1. **Hard Constraint Filtering**
   - Age compatibility: `child_age ∈ [bucket.min_age, bucket.max_age]`
   - Distance: `distance(home, center) ≤ max_distance`  
   - Hours overlap: `desired_hours ∩ opening_hours ≠ ∅`
   - Must-have preferences: All threshold ≥ 0.9 satisfied
   - Exclusions: No threshold ≤ 0.1 matched
   - Reserved buckets: Required priority flags present

2. **Soft Scoring**
   ```python
   weight = α·preference_score + β·property_score + γ·availability_score + δ·quality_score - ε·distance_penalty + ζ·sibling_bonus
   ```

### Graph Pruning

To maintain performance:
- **Edge Limit**: Max 50 edges per application  
- **Weight Threshold**: Remove edges with weight < 0.1
- **Distance Sorting**: Prioritize nearby centers first

## Algorithm 1: Recommend Mode (Top-K)

**Use Case**: Generate personalized recommendations for a parent

### Algorithm
```python
def recommend(application_id, k=10):
    app_node = f"app_{application_id}"
    
    # Get all edges from this application
    edges = [(neighbor, weight) for neighbor in graph[app_node]]
    
    # Sort by weight descending
    edges.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return edges[:k]
```

### Complexity
- **Time**: O(E_app log E_app) where E_app = edges from application
- **Space**: O(E_app)
- **Typical**: ~50 edges per app → very fast

### Example
```python
# Input: Parent wants organic food, max 3km
application = {
  "preferences": [{"organic_food": 0.9}],
  "max_distance_km": 3.0
}

# Output: Ranked recommendations
[
  {"center": "Organic Garden", "score": 0.92, "distance": 1.2},
  {"center": "Green Sprouts", "score": 0.87, "distance": 2.1},
  {"center": "Natural Learning", "score": 0.81, "distance": 2.8}
]
```

## Algorithm 2: Allocate Mode (Min-Cost Flow)

**Use Case**: Optimal global allocation respecting capacity constraints

### Problem Formulation

**Objective**: Maximize total weighted matches
```
maximize: Σ w_ij * x_ij
```

**Subject to**:
- Each application matched at most once: `Σ_j x_ij ≤ 1 ∀i`
- Capacity constraints: `Σ_i x_ij ≤ capacity_j ∀j`  
- Binary variables: `x_ij ∈ {0,1}`

### Min-Cost Flow Transformation

Convert to min-cost flow by:
1. **Source** → Applications (capacity 1, cost 0)
2. **Applications** → Buckets (capacity 1, cost = -weight)  
3. **Buckets** → Sink (capacity = bucket_capacity, cost 0)

```
Source ──→ App1 ──→ Bucket1 ──→ Sink
   │       │   ╲     ╱    │
   │       │    ╲   ╱     │ 
   │       │     ╲ ╱      │
   └──→ App2 ──→ Bucket2 ──┘
```

### Implementation
```python
def allocate_optimal(applications, centers):
    # Build flow network
    mcf = SimpleMinCostFlow()
    
    # Add source → applications
    for i, app in enumerate(applications):
        mcf.add_arc(SOURCE, app_node(i), 1, 0)
    
    # Add applications → buckets  
    for i, app in enumerate(applications):
        for j, bucket in enumerate(buckets):
            if is_compatible(app, bucket):
                weight = score(app, bucket)
                cost = int(-weight * 1000)  # Negative for maximization
                mcf.add_arc(app_node(i), bucket_node(j), 1, cost)
    
    # Add buckets → sink
    for j, bucket in enumerate(buckets):
        mcf.add_arc(bucket_node(j), SINK, bucket.capacity, 0)
    
    # Solve
    status = mcf.solve()
    return extract_matches(mcf)
```

### Complexity
- **Time**: O(V³) for min-cost flow, where V = applications + buckets
- **Space**: O(V + E)
- **Scalability**: ~1000 applications, ~500 centers → ~2 seconds

### Sibling Handling

Post-processing for sibling co-assignment:
```python
def adjust_for_siblings(matches):
    families = group_by_family(matches)
    
    for family_matches in families:
        if len(family_matches) > 1:
            centers = [m.center_id for m in family_matches]
            
            if len(set(centers)) > 1:  # Split family
                # Try to consolidate at best center
                best_center = most_common_center(centers)
                reallocate_family(family_matches, best_center)
```

## Algorithm 3: Waitlist Mode (Tiered Sorting)

**Use Case**: Rank all applications interested in a specific center

### Algorithm
```python
def generate_waitlist(center_id, policy_tiers):
    # Find all applications → center edges
    candidates = []
    
    for bucket in center.buckets:
        for app_node in graph.neighbors(bucket_node):
            app = get_application(app_node)
            weight = graph[app_node][bucket_node]['weight']
            
            # Apply policy tier bonus
            tier_bonus = calculate_tier_bonus(app, policy_tiers)
            adjusted_weight = weight + tier_bonus
            
            candidates.append({
                'application': app,
                'weight': adjusted_weight,
                'tier': determine_tier(app, policy_tiers)
            })
    
    # Sort by adjusted weight (tier bonus included)
    candidates.sort(key=lambda x: x['weight'], reverse=True)
    
    return candidates
```

### Policy Tiers

Hierarchical priority system:
```python
POLICY_TIERS = {
    'sibling': 1000,        # Existing student's sibling
    'municipality': 800,    # Lives in municipality
    'staff_children': 600,  # Staff member's child
    'low_income': 500,      # Income-qualified
    'special_needs': 300,   # Special accommodations
    'regular': 0            # No special priority
}
```

### Tie-Breaking
1. **Policy tier** (highest priority first)
2. **Match score** (higher score first) 
3. **Application date** (earlier application first)
4. **Random seed** (deterministic randomness)

### Example Output
```python
[
  {"rank": 1, "app": "family_a", "tier": "sibling", "score": 0.95},
  {"rank": 2, "app": "family_b", "tier": "sibling", "score": 0.91},
  {"rank": 3, "app": "family_c", "tier": "municipality", "score": 0.88},
  {"rank": 4, "app": "family_d", "tier": "regular", "score": 0.94}
]
```

Note: Regular family_d has higher base score but ranks lower due to tier.

## Performance Optimizations

### 1. Graph Pruning
```python
# Limit edges per application
max_edges_per_app = 50

# Remove low-weight edges  
min_edge_weight = 0.1

# Pre-filter by distance
distance_radius = 20.0  # km
```

### 2. Parallel Processing
```python
# Score edges in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(score_edge, app, bucket) 
              for app, bucket in candidate_pairs]
    
    weights = [f.result() for f in futures]
```

### 3. Caching
- **Graph structure**: Cache for identical center sets
- **Scoring results**: Cache preference evaluations  
- **Distance calculations**: Cache geographic distances

### 4. Progressive Refinement
```python
def progressive_matching(application):
    batch_size = 100
    quality_matches = []
    
    while len(quality_matches) < target_matches:
        centers = load_next_batch(batch_size)
        if not centers:
            break
            
        matches = quick_match(application, centers)
        quality_matches.extend(filter_quality(matches))
        
        batch_size *= 2  # Exponential expansion
```

## Algorithm Comparison

| Aspect | Recommend | Allocate | Waitlist |
|--------|-----------|----------|----------|
| **Scope** | Single app | All apps | Single center |
| **Objective** | Top-K ranking | Global optimum | Fair ordering |
| **Complexity** | O(E log E) | O(V³) | O(E log E) |
| **Constraints** | Soft only | Hard + soft | Policy tiers |
| **Output** | Recommendations | Assignments | Rankings |
| **Use Case** | Parent search | Batch allocation | Waitlist mgmt |

## Future Enhancements

### 1. Stable Marriage Algorithm
For preference-based mutual matching:
```python
def stable_marriage_matching(applications, centers):
    # Applications and centers both have preferences
    # Find stable assignment where no pair wants to defect
```

### 2. Multi-Objective Optimization  
Balance competing objectives:
- **Efficiency**: Maximize total satisfaction
- **Fairness**: Minimize satisfaction variance
- **Utilization**: Maximize capacity utilization

### 3. Online Algorithms
Handle dynamic arrivals/departures:
- **Incremental updates**: Add/remove without full recomputation
- **Approximate algorithms**: Trade optimality for speed
- **Learning algorithms**: Adapt based on historical data

### 4. Machine Learning Integration
```python
def ml_enhanced_scoring(application, center):
    features = extract_features(application, center)
    ml_score = model.predict_acceptance_probability(features)
    
    # Combine rule-based and ML scores
    return alpha * rule_score + (1-alpha) * ml_score
```