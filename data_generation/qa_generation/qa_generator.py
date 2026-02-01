"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.
Author: Jesse Pokora

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
Physics QA Generator

Generates diverse question-answer pairs from physics simulations.
Supports multiple question types for comprehensive physics reasoning.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random


class QuestionType(Enum):
    # ==========================================================================
    # CATEGORY 1: Basic Physical Properties (Perceptual Grounding)
    # These questions test understanding of directly observable physical properties
    # ==========================================================================
    OBJECT_COUNT = "object_count"  # How many objects are present?
    OBJECT_POSITION = "object_position"  # Where is object X located?
    OBJECT_VELOCITY = "object_velocity"  # What is the velocity of object X?
    OBJECT_MASS = "object_mass"  # What is the relative mass of object X?
    MOTION_DIRECTION = "motion_direction"  # In which direction is object X moving?
    
    # ==========================================================================
    # CATEGORY 2: Physical Quantities (Numerical Reasoning)
    # These questions require computing physical quantities from state
    # ==========================================================================
    KINETIC_ENERGY = "kinetic_energy"  # What is the kinetic energy of the system?
    TOTAL_MOMENTUM = "total_momentum"  # What is the total momentum?
    RELATIVE_VELOCITY = "relative_velocity"  # What is the relative velocity between objects?
    SPATIAL_DISTANCE = "spatial_distance"  # What is the distance between objects?
    SPEED_COMPARISON = "speed_comparison"  # Which object is moving fastest?
    MASS_COMPARISON = "mass_comparison"  # Which object has the greatest mass?
    
    # ==========================================================================
    # CATEGORY 3: Predictive Reasoning (Temporal Inference)
    # These questions require predicting future states from current dynamics
    # ==========================================================================
    COLLISION_PREDICTION = "collision_prediction"  # Will these objects collide?
    TRAJECTORY_EXTRAPOLATION = "trajectory_extrapolation"  # Where will object X be?
    TIME_TO_EVENT = "time_to_event"  # When will event X occur?
    REACHABILITY = "reachability"  # Can object X reach location Y?
    PATH_OBSTRUCTION = "path_obstruction"  # Is the path between X and Y obstructed?
    
    # ==========================================================================
    # CATEGORY 4: Relational Reasoning (Spatial Relations)
    # These questions test understanding of spatial relationships
    # ==========================================================================
    PROXIMITY = "proximity"  # Which object is nearest to X?
    SPATIAL_CONTAINMENT = "spatial_containment"  # How many objects are within region R?
    RELATIVE_POSITION = "relative_position"  # Is X above/below/left/right of Y?
    CONTACT_STATE = "contact_state"  # Is object X in contact with a surface?
    
    # ==========================================================================
    # CATEGORY 5: Evaluative Concepts (Safety & Normative Judgments)
    # These ground abstract evaluative concepts in physical states
    # Key insight: "safe", "dangerous", "threatening" are physically grounded
    # ==========================================================================
    SAFETY_ASSESSMENT = "safety_assessment"  # Is this situation safe?
    DANGER_LEVEL = "danger_level"  # How dangerous is this configuration?
    THREAT_TO_SELF = "threat_to_self"  # Is the agent in danger?
    THREAT_TO_OTHERS = "threat_to_others"  # Is the agent endangering others?
    STABILITY_ASSESSMENT = "stability_assessment"  # Is this configuration stable?
    COLLISION_RISK = "collision_risk"  # What is the risk of collision?
    ESCAPE_ROUTES = "escape_routes"  # Are there safe paths available?
    PROTECTIVE_ACTION = "protective_action"  # What action would increase safety?
    VULNERABILITY = "vulnerability"  # Which objects are most vulnerable?
    CAUSAL_RESPONSIBILITY = "causal_responsibility"  # Who/what would cause harm?
    FORCE_ASSESSMENT = "force_assessment"  # Is this too much/little force? (gentle/firm/crushing)
    STRUCTURAL_LOAD = "structural_load"  # Can this support that weight?
    URGENCY_ASSESSMENT = "urgency_assessment"  # How urgent is action required?
    
    # ==========================================================================
    # CATEGORY 6: Intentional Concepts (Agency & Goal Attribution)
    # These ground concepts of agency, intention, and purpose in motion patterns
    # ==========================================================================
    AGENT_IDENTIFICATION = "agent_identification"  # Which objects appear to be agents?
    GOAL_INFERENCE = "goal_inference"  # What is this object trying to do?
    HELPING_HINDERING = "helping_hindering"  # Is object X helping or hindering Y?
    CHASING_FLEEING = "chasing_fleeing"  # Is X chasing Y or fleeing from Y?
    COOPERATION_COMPETITION = "cooperation_competition"  # Are objects cooperating or competing?
    
    METAPHOR_COLLISION = "metaphor_collision"
    METAPHOR_MOMENTUM = "metaphor_momentum"
    METAPHOR_EQUILIBRIUM = "metaphor_equilibrium"
    METAPHOR_TRAJECTORY = "metaphor_trajectory"
    METAPHOR_FORCE = "metaphor_force"
    
    # New Lakoff-aligned metaphor types (from "Metaphors We Live By")
    METAPHOR_CONTAINER = "metaphor_container"  # CONTAINMENT schema
    METAPHOR_SOURCE_PATH_GOAL = "metaphor_source_path_goal"  # SOURCE-PATH-GOAL schema
    METAPHOR_BALANCE = "metaphor_balance"  # BALANCE schema
    METAPHOR_LINK = "metaphor_link"  # LINK schema
    METAPHOR_CENTER_PERIPHERY = "metaphor_center_periphery"  # CENTER-PERIPHERY schema
    METAPHOR_RESISTANCE = "metaphor_resistance"  # BLOCKAGE/RESISTANCE schema
    
    # Mathematical metaphors from "Where Mathematics Comes From" (Lakoff & Núñez)
    METAPHOR_ARITHMETIC_MOTION = "metaphor_arithmetic_motion"  # Arithmetic is motion along a path
    METAPHOR_ARITHMETIC_COLLECTION = "metaphor_arithmetic_collection"  # Arithmetic is object collection
    METAPHOR_ARITHMETIC_CONSTRUCTION = "metaphor_arithmetic_construction"  # Arithmetic is object construction
    METAPHOR_MEASURING_STICK = "metaphor_measuring_stick"  # Numbers as measuring stick segments
    METAPHOR_SETS_CONTAINERS = "metaphor_sets_containers"  # Sets are containers
    METAPHOR_CONTINUITY_GAPLESS = "metaphor_continuity_gapless"  # Continuity is gapless motion
    METAPHOR_CHANGE_MOTION = "metaphor_change_motion"  # Change is motion
    METAPHOR_NUMBERS_POINTS = "metaphor_numbers_points"  # Numbers are points on a line
    METAPHOR_RECURRENCE_CIRCULAR = "metaphor_recurrence_circular"  # Recurrence/cycles are circular motion
    METAPHOR_INFINITY = "metaphor_infinity"  # Basic Metaphor of Infinity (BMI)
    
    # ==========================================================================
    # CATEGORY 7: CLEVRER-Style Reasoning (Benchmark Alignment)
    # These question types align with the CLEVRER benchmark for evaluation
    # ==========================================================================
    CAUSAL_CHAIN = "causal_chain"  # What caused event X? (explanatory)
    FUTURE_PREDICTION = "future_prediction"  # What will happen next? (predictive)
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"  # What if X were removed? (counterfactual)


@dataclass
class QAPair:
    states: torch.Tensor
    mask: torch.Tensor
    question: str
    answer: str
    question_type: QuestionType
    metadata: Dict


class PhysicsQAGenerator:
    """
    Generates diverse QA pairs from physics simulations.
    
    Question types:
    - Counting: "How many objects are in the scene?"
    - Velocity: "How many objects are moving?"
    - Direction: "How many objects are moving left/right/up/down?"
    - Collision: "Will any objects collide?"
    - Energy: "What is the total kinetic energy?"
    - Momentum: "What is the total momentum?"
    - Position: "Where is the largest object?"
    - Comparison: "Which object is moving fastest?"
    """
    
    POSITION_IDX = slice(0, 3)  # x, y, z
    VELOCITY_IDX = slice(3, 6)  # vx, vy, vz
    MASS_IDX = 6  # mass
    RADIUS_IDX = 7  # radius
    
    def __init__(
        self,
        question_types: Optional[List[QuestionType]] = None,
        position_change_threshold: float = 0.1,
        collision_threshold: float = 3.0,
        seed: int = 42
    ):
        """
        Args:
            question_types: List of question types to generate. None = all types.
            position_change_threshold: Minimum position change to consider object "moving"
            collision_threshold: Distance threshold for collision prediction
            seed: Random seed for reproducibility
        """
        self.question_types = question_types or list(QuestionType)
        self.position_change_threshold = position_change_threshold
        self.collision_threshold = collision_threshold
        self.rng = random.Random(seed)
        
        self.question_templates = {
            # =================================================================
            # CATEGORY 1: Basic Physical Properties (Perceptual Grounding)
            # =================================================================
            QuestionType.OBJECT_COUNT: [
                ("How many objects are present in this physical system?", self._answer_counting),
                ("Count the number of objects in the scene.", self._answer_counting),
                ("What is the total object count?", self._answer_counting),
            ],
            QuestionType.OBJECT_POSITION: [
                ("What is the position of the largest object?", self._answer_largest_position),
                ("Describe the spatial location of the primary object.", self._answer_largest_position),
                ("Where is the most massive object located?", self._answer_largest_position),
            ],
            QuestionType.OBJECT_VELOCITY: [
                ("How many objects have non-zero velocity?", self._answer_moving_count),
                ("Count the objects that are in motion.", self._answer_moving_count),
                ("How many objects are currently moving?", self._answer_moving_count),
            ],
            QuestionType.OBJECT_MASS: [
                ("Which object has the largest mass?", self._answer_heaviest_object_num),
                ("Identify the most massive object.", self._answer_heaviest_object_num),
                ("What is the index of the heaviest object?", self._answer_heaviest_object_num),
            ],
            QuestionType.MOTION_DIRECTION: [
                ("How many objects are moving rightward?", lambda s, m: self._answer_direction_count(s, m, 'right')),
                ("How many objects are moving leftward?", lambda s, m: self._answer_direction_count(s, m, 'left')),
                ("How many objects are moving upward?", lambda s, m: self._answer_direction_count(s, m, 'up')),
                ("How many objects are moving downward?", lambda s, m: self._answer_direction_count(s, m, 'down')),
            ],
            
            # =================================================================
            # CATEGORY 2: Physical Quantities (Numerical Reasoning)
            # =================================================================
            QuestionType.KINETIC_ENERGY: [
                ("What is the total kinetic energy of the system?", self._answer_kinetic_energy_calc),
                ("Calculate the sum of kinetic energies.", self._answer_kinetic_energy_calc),
                ("What is the system's total kinetic energy?", self._answer_kinetic_energy_calc),
            ],
            QuestionType.TOTAL_MOMENTUM: [
                ("What is the total momentum magnitude of the system?", self._answer_momentum_calc),
                ("Calculate the net momentum.", self._answer_momentum_calc),
                ("What is the vector sum of all momenta?", self._answer_momentum_calc),
            ],
            QuestionType.RELATIVE_VELOCITY: [
                ("Are the two nearest objects approaching or separating?", self._answer_relative_motion),
                ("What is the relative motion between the closest pair?", self._answer_relative_motion),
                ("Is the distance between the nearest objects increasing or decreasing?", self._answer_relative_motion),
            ],
            QuestionType.SPATIAL_DISTANCE: [
                ("What is the minimum distance between any two objects?", self._answer_min_distance_calc),
                ("How close are the nearest pair of objects?", self._answer_min_distance_calc),
                ("What is the smallest inter-object distance?", self._answer_min_distance_calc),
            ],
            QuestionType.SPEED_COMPARISON: [
                ("Which object has the highest speed?", self._answer_fastest_object_num),
                ("Identify the fastest moving object.", self._answer_fastest_object_num),
                ("What is the maximum speed in the system?", self._answer_max_speed_calc),
            ],
            QuestionType.MASS_COMPARISON: [
                ("Which object has the greatest mass?", self._answer_heaviest_object_num),
                ("Compare the masses: which object is heaviest?", self._answer_heaviest_object_num),
                ("Identify the most massive object.", self._answer_heaviest_object_num),
            ],
            
            # =================================================================
            # CATEGORY 3: Predictive Reasoning (Temporal Inference)
            # =================================================================
            QuestionType.COLLISION_PREDICTION: [
                ("Will any objects collide in the near future?", self._answer_collision),
                ("Is a collision imminent between any pair of objects?", self._answer_collision),
                ("Predict whether a collision will occur.", self._answer_will_collide_soon),
            ],
            QuestionType.TRAJECTORY_EXTRAPOLATION: [
                ("Where will object 1 be after 10 time steps?", lambda s, m: self._answer_trajectory_prediction(s, m, 0, 10)),
                ("Extrapolate the position of the primary object.", lambda s, m: self._answer_trajectory_prediction(s, m, 0, 10)),
                ("Predict the future location of object 2.", lambda s, m: self._answer_trajectory_prediction(s, m, 1, 10)),
            ],
            QuestionType.TIME_TO_EVENT: [
                ("How many time steps until the nearest pair collides?", self._answer_time_to_collision),
                ("Estimate the time to collision.", self._answer_time_to_collision),
                ("When will the closest objects make contact?", self._answer_time_to_collision),
            ],
            QuestionType.REACHABILITY: [
                ("Can object 1 reach the target position before object 2?", self._answer_can_reach_target),
                ("Which object will reach the destination first?", self._answer_can_reach_target),
                ("Is the target reachable by the primary object?", self._answer_can_reach_target),
            ],
            QuestionType.PATH_OBSTRUCTION: [
                ("Is the direct path between object 1 and object 3 obstructed?", self._answer_is_path_blocked),
                ("Are there obstacles between the source and destination?", self._answer_is_path_blocked),
                ("Is line-of-sight blocked between the two objects?", self._answer_is_path_blocked),
            ],
            
            # =================================================================
            # CATEGORY 4: Relational Reasoning (Spatial Relations)
            # =================================================================
            QuestionType.PROXIMITY: [
                ("Which object is nearest to object 1?", lambda s, m: self._answer_nearest_object(s, m, 0)),
                ("Identify the closest neighbor of the primary object.", lambda s, m: self._answer_nearest_object(s, m, 0)),
                ("What is the nearest object to the reference point?", lambda s, m: self._answer_nearest_object(s, m, 0)),
            ],
            QuestionType.SPATIAL_CONTAINMENT: [
                ("How many objects are within 2.0 units of object 1?", lambda s, m: self._answer_objects_in_range(s, m, 0, 2.0)),
                ("Count objects within a 3.0 unit radius of the reference.", lambda s, m: self._answer_objects_in_range(s, m, 0, 3.0)),
                ("How many objects fall within the specified region?", lambda s, m: self._answer_objects_in_range(s, m, 0, 2.0)),
            ],
            QuestionType.RELATIVE_POSITION: [
                ("Describe the relative positions of the objects.", self._answer_largest_position),
                ("What is the spatial arrangement of the system?", self._answer_largest_position),
            ],
            QuestionType.CONTACT_STATE: [
                ("Is object 1 in contact with a surface?", lambda s, m: self._answer_is_grounded(s, m, 0)),
                ("Is the primary object resting on a boundary?", lambda s, m: self._answer_is_grounded(s, m, 0)),
                ("Determine if object 1 is grounded.", lambda s, m: self._answer_is_grounded(s, m, 0)),
            ],
            
            # =================================================================
            # CATEGORY 5: Evaluative Concepts (Safety & Normative Judgments)
            # =================================================================
            QuestionType.SAFETY_ASSESSMENT: [
                ("Is this situation safe?", self._answer_safety_assessment),
                ("Assess the overall safety of this configuration.", self._answer_safety_assessment),
                ("Would you consider this environment safe to operate in?", self._answer_safety_assessment),
            ],
            QuestionType.DANGER_LEVEL: [
                ("How dangerous is this situation?", self._answer_danger_level),
                ("Rate the danger level of this configuration.", self._answer_danger_level),
                ("On a scale from safe to critical, how would you assess this?", self._answer_danger_level),
            ],
            QuestionType.THREAT_TO_SELF: [
                ("Is the primary agent in danger?", self._answer_threat_to_self),
                ("Is object 1 at risk of being harmed?", self._answer_threat_to_self),
                ("Should the agent take evasive action?", self._answer_threat_to_self),
            ],
            QuestionType.THREAT_TO_OTHERS: [
                ("Is the agent endangering other objects?", self._answer_threat_to_others),
                ("Could the primary object's motion harm others?", self._answer_threat_to_others),
                ("Is object 1 on a collision course with vulnerable objects?", self._answer_threat_to_others),
            ],
            QuestionType.STABILITY_ASSESSMENT: [
                ("Is this configuration stable?", self._answer_stability_assessment),
                ("Will this arrangement remain balanced?", self._answer_stability_assessment),
                ("Assess the structural stability of this system.", self._answer_stability_assessment),
            ],
            QuestionType.COLLISION_RISK: [
                ("What is the collision risk in this scenario?", self._answer_collision_risk),
                ("How likely is a collision to occur?", self._answer_collision_risk),
                ("Assess the probability of impact.", self._answer_collision_risk),
            ],
            QuestionType.ESCAPE_ROUTES: [
                ("Are there safe paths available for escape?", self._answer_escape_routes),
                ("Can the agent move to safety?", self._answer_escape_routes),
                ("Identify available escape routes.", self._answer_escape_routes),
            ],
            QuestionType.PROTECTIVE_ACTION: [
                ("What action would increase safety?", self._answer_protective_action),
                ("How should the agent respond to reduce risk?", self._answer_protective_action),
                ("What protective measure is recommended?", self._answer_protective_action),
            ],
            QuestionType.VULNERABILITY: [
                ("Which objects are most vulnerable?", self._answer_vulnerability),
                ("Identify the most at-risk elements in this scene.", self._answer_vulnerability),
                ("What is most likely to be damaged?", self._answer_vulnerability),
            ],
            QuestionType.CAUSAL_RESPONSIBILITY: [
                ("If harm occurs, what would be the cause?", self._answer_causal_responsibility),
                ("Which object would be responsible for a collision?", self._answer_causal_responsibility),
                ("Identify the primary source of danger.", self._answer_causal_responsibility),
            ],
            QuestionType.FORCE_ASSESSMENT: [
                ("Is this interaction gentle or forceful?", self._answer_force_assessment),
                ("Assess the force level: gentle, moderate, or crushing?", self._answer_force_assessment),
                ("How much force is being applied in this interaction?", self._answer_force_assessment),
            ],
            QuestionType.STRUCTURAL_LOAD: [
                ("Can this configuration support the weight?", self._answer_structural_load),
                ("Is this structure load-bearing capacity sufficient?", self._answer_structural_load),
                ("Will this arrangement hold or collapse?", self._answer_structural_load),
            ],
            QuestionType.URGENCY_ASSESSMENT: [
                ("How urgent is action required?", self._answer_urgency_assessment),
                ("Is immediate response needed or can we wait?", self._answer_urgency_assessment),
                ("Assess the time pressure of this situation.", self._answer_urgency_assessment),
            ],
            
            # =================================================================
            # CATEGORY 6: Intentional Concepts (Agency & Goal Attribution)
            # =================================================================
            QuestionType.AGENT_IDENTIFICATION: [
                ("Which objects appear to be agents with goals?", self._answer_agent_identification),
                ("Identify objects that seem to be acting purposefully.", self._answer_agent_identification),
                ("Which objects show intentional behavior?", self._answer_agent_identification),
            ],
            QuestionType.GOAL_INFERENCE: [
                ("What is object 1 trying to achieve?", self._answer_goal_inference),
                ("Infer the goal of the primary moving object.", self._answer_goal_inference),
                ("What appears to be the intention behind this motion?", self._answer_goal_inference),
            ],
            QuestionType.HELPING_HINDERING: [
                ("Is object 2 helping or hindering object 1?", self._answer_helping_hindering),
                ("Assess whether the interaction is cooperative or obstructive.", self._answer_helping_hindering),
                ("Is this object facilitating or blocking progress?", self._answer_helping_hindering),
            ],
            QuestionType.CHASING_FLEEING: [
                ("Is object 1 chasing object 2, or fleeing from it?", self._answer_chasing_fleeing),
                ("Characterize the pursuit dynamics between these objects.", self._answer_chasing_fleeing),
                ("Is this a chase or an escape?", self._answer_chasing_fleeing),
            ],
            QuestionType.COOPERATION_COMPETITION: [
                ("Are these objects cooperating or competing?", self._answer_cooperation_competition),
                ("Is this interaction collaborative or adversarial?", self._answer_cooperation_competition),
                ("Assess the social dynamics of this scene.", self._answer_cooperation_competition),
            ],
            
            QuestionType.METAPHOR_COLLISION: [
                ("If these objects were people in a debate, what would happen when they meet?", self._answer_metaphor_collision),
                ("Describe this collision as if it were a conflict between ideas.", self._answer_metaphor_collision),
                ("If object 1 were an argument and object 2 were a counterargument, what happens?", self._answer_metaphor_collision),
            ],
            QuestionType.METAPHOR_MOMENTUM: [
                ("If this object's momentum represented career progress, describe it.", self._answer_metaphor_momentum),
                ("Describe this motion as if it were someone's emotional state.", self._answer_metaphor_momentum),
                ("If the momentum were political influence, what would you say?", self._answer_metaphor_momentum),
            ],
            QuestionType.METAPHOR_EQUILIBRIUM: [
                ("If this system were a relationship, is it balanced or unstable?", self._answer_metaphor_equilibrium),
                ("Describe the system's balance as if it were a work-life situation.", self._answer_metaphor_equilibrium),
                ("If these forces were competing priorities, what's the outcome?", self._answer_metaphor_equilibrium),
            ],
            QuestionType.METAPHOR_TRAJECTORY: [
                ("If this trajectory were a person's life path, where are they heading?", self._answer_metaphor_trajectory),
                ("Describe this motion as if it were a company's growth trajectory.", self._answer_metaphor_trajectory),
                ("If this path represented a student's academic journey, what would you predict?", self._answer_metaphor_trajectory),
            ],
            QuestionType.METAPHOR_FORCE: [
                ("If the forces on this object were social pressures, describe the situation.", self._answer_metaphor_force),
                ("Describe the net force as if it were peer pressure on a decision.", self._answer_metaphor_force),
                ("If these forces were competing influences on a policy, what's happening?", self._answer_metaphor_force),
            ],
            QuestionType.METAPHOR_CONTAINER: [
                ("If this region were a container for ideas, describe what's inside vs outside.", self._answer_metaphor_container),
                ("Describe this spatial arrangement as if objects were thoughts entering or leaving your mind.", self._answer_metaphor_container),
                ("If the scene boundary were the limits of a project scope, what's included?", self._answer_metaphor_container),
            ],
            QuestionType.METAPHOR_SOURCE_PATH_GOAL: [
                ("Describe this motion as a journey from origin to destination.", self._answer_metaphor_source_path_goal),
                ("If this trajectory were a project timeline, where did it start and where is it heading?", self._answer_metaphor_source_path_goal),
                ("Map this movement to a personal transformation journey.", self._answer_metaphor_source_path_goal),
            ],
            QuestionType.METAPHOR_BALANCE: [
                ("If this system's balance represented justice, is it fair?", self._answer_metaphor_balance),
                ("Describe the weight distribution as if it were resource allocation in an organization.", self._answer_metaphor_balance),
                ("If the masses were responsibilities, who carries the most burden?", self._answer_metaphor_balance),
            ],
            QuestionType.METAPHOR_LINK: [
                ("Describe the connections between objects as if they were relationships between people.", self._answer_metaphor_link),
                ("If proximity represented trust, who trusts whom?", self._answer_metaphor_link),
                ("Map the spatial relationships to a social network.", self._answer_metaphor_link),
            ],
            QuestionType.METAPHOR_CENTER_PERIPHERY: [
                ("If position represented importance, who is central and who is marginalized?", self._answer_metaphor_center_periphery),
                ("Describe the spatial arrangement as organizational hierarchy.", self._answer_metaphor_center_periphery),
                ("If the center were power, who has influence?", self._answer_metaphor_center_periphery),
            ],
            QuestionType.METAPHOR_RESISTANCE: [
                ("Describe any obstacles or resistance in this scene as if they were challenges in achieving a goal.", self._answer_metaphor_resistance),
                ("If slower objects represented bureaucratic friction, what's the situation?", self._answer_metaphor_resistance),
                ("Map the impediments to progress as organizational barriers.", self._answer_metaphor_resistance),
            ],
            # Mathematical metaphors from "Where Mathematics Comes From"
            QuestionType.METAPHOR_ARITHMETIC_MOTION: [
                ("If the motion along this path represented addition or subtraction, what arithmetic is happening?", self._answer_metaphor_arithmetic_motion),
                ("Describe the movement as walking along a number line.", self._answer_metaphor_arithmetic_motion),
                ("Map this trajectory to arithmetic operations: how many steps forward or backward?", self._answer_metaphor_arithmetic_motion),
            ],
            QuestionType.METAPHOR_ARITHMETIC_COLLECTION: [
                ("If these objects were being collected into groups, what addition is happening?", self._answer_metaphor_arithmetic_collection),
                ("Describe the grouping of objects as arithmetic with collections.", self._answer_metaphor_arithmetic_collection),
                ("How would you express this scene as putting objects together or taking them apart?", self._answer_metaphor_arithmetic_collection),
            ],
            QuestionType.METAPHOR_ARITHMETIC_CONSTRUCTION: [
                ("If objects were being constructed or deconstructed, what multiplication or division is represented?", self._answer_metaphor_arithmetic_construction),
                ("Describe building or breaking down objects as arithmetic operations.", self._answer_metaphor_arithmetic_construction),
                ("How does the construction/destruction of objects map to multiplication?", self._answer_metaphor_arithmetic_construction),
            ],
            QuestionType.METAPHOR_MEASURING_STICK: [
                ("If distances were measured in unit lengths, what numbers do you see?", self._answer_metaphor_measuring_stick),
                ("Describe the spatial relationships as measurements on a ruler.", self._answer_metaphor_measuring_stick),
                ("How many unit lengths separate the key objects?", self._answer_metaphor_measuring_stick),
            ],
            QuestionType.METAPHOR_SETS_CONTAINERS: [
                ("If spatial regions were sets, which objects belong to which set?", self._answer_metaphor_sets_containers),
                ("Describe the containment relationships as set membership.", self._answer_metaphor_sets_containers),
                ("What is the intersection and union of the object groupings?", self._answer_metaphor_sets_containers),
            ],
            QuestionType.METAPHOR_CONTINUITY_GAPLESS: [
                ("Is the motion continuous or are there gaps? Describe as a mathematical function.", self._answer_metaphor_continuity_gapless),
                ("If this trajectory were a function, is it continuous or discontinuous?", self._answer_metaphor_continuity_gapless),
                ("Describe the smoothness or jumpiness of motion as continuity.", self._answer_metaphor_continuity_gapless),
            ],
            QuestionType.METAPHOR_CHANGE_MOTION: [
                ("Describe the rate of change as speed of motion.", self._answer_metaphor_change_motion),
                ("If change were motion, how fast is this system changing?", self._answer_metaphor_change_motion),
                ("Map the derivatives (rates of change) to velocities in this scene.", self._answer_metaphor_change_motion),
            ],
            QuestionType.METAPHOR_NUMBERS_POINTS: [
                ("If positions were numbers on a number line, what values do you see?", self._answer_metaphor_numbers_points),
                ("Map the spatial positions to points on a coordinate system.", self._answer_metaphor_numbers_points),
                ("Describe the objects as numbers located at specific points.", self._answer_metaphor_numbers_points),
            ],
            QuestionType.METAPHOR_RECURRENCE_CIRCULAR: [
                ("Is there any cyclical or recurring pattern? Describe it as circular motion.", self._answer_metaphor_recurrence_circular),
                ("If repetition were circular, what cycles do you observe?", self._answer_metaphor_recurrence_circular),
                ("Map any periodic behavior to rotation or orbiting.", self._answer_metaphor_recurrence_circular),
            ],
            QuestionType.METAPHOR_INFINITY: [
                ("If this motion continued forever, what would the limit be?", self._answer_metaphor_infinity),
                ("Describe the potential infinite continuation of this process.", self._answer_metaphor_infinity),
                ("What happens as we extend this pattern toward infinity?", self._answer_metaphor_infinity),
            ],
            
            # =================================================================
            # CATEGORY 7: CLEVRER-Style Reasoning (Benchmark Alignment)
            # =================================================================
            QuestionType.CAUSAL_CHAIN: [
                ("What caused the current motion pattern?", self._answer_causal_chain),
                ("Explain the causal sequence leading to this state.", self._answer_causal_chain),
                ("What events led to this configuration?", self._answer_causal_chain),
            ],
            QuestionType.FUTURE_PREDICTION: [
                ("What will happen next in this scene?", self._answer_future_prediction),
                ("Predict the next significant event.", self._answer_future_prediction),
                ("What outcome do you expect from this configuration?", self._answer_future_prediction),
            ],
            QuestionType.COUNTERFACTUAL_REASONING: [
                ("What would happen if object 1 were removed?", self._answer_counterfactual),
                ("How would the outcome change without the fastest object?", self._answer_counterfactual),
                ("If the largest object weren't present, what would be different?", self._answer_counterfactual),
            ],
        }
    
    def generate_qa_pair(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        question_type: Optional[QuestionType] = None
    ) -> QAPair:
        """
        Generate a single QA pair from physics states.
        
        Args:
            states: [seq_len, objects, state_dim] or [objects, state_dim]
            mask: [seq_len, objects] or [objects]
            question_type: Specific type or None for random
            
        Returns:
            QAPair with question, answer, and metadata
        """
        if question_type is None:
            _, _, _, _, count = self._get_active_objects(states, mask)
            applicable_types = []
            for qt in self.question_types:
                if qt == QuestionType.PATH_OBSTRUCTION and count < 3:
                    continue
                if qt == QuestionType.REACHABILITY and count < 2:
                    continue
                if qt == QuestionType.SPATIAL_CONTAINMENT and count < 2:
                    continue
                applicable_types.append(qt)

            if not applicable_types:
                applicable_types = [QuestionType.OBJECT_COUNT]

            question_type = self.rng.choice(applicable_types)
        
        templates = self.question_templates[question_type]
        question_template, answer_fn = self.rng.choice(templates)
        
        answer, metadata = answer_fn(states, mask)
        
        return QAPair(
            states=states,
            mask=mask,
            question=question_template,
            answer=answer,
            question_type=question_type,
            metadata=metadata
        )
    
    def generate_batch(
        self,
        states_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        num_per_sample: int = 1
    ) -> List[QAPair]:
        """
        Generate QA pairs for a batch of physics simulations.
        
        Args:
            states_batch: [batch, seq_len, objects, state_dim]
            mask_batch: [batch, seq_len, objects]
            num_per_sample: Number of QA pairs per sample
            
        Returns:
            List of QAPairs
        """
        qa_pairs = []
        batch_size = states_batch.size(0)
        
        for i in range(batch_size):
            for _ in range(num_per_sample):
                qa = self.generate_qa_pair(states_batch[i], mask_batch[i])
                qa_pairs.append(qa)
        
        return qa_pairs
    
    def _get_active_objects(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get states of active objects and compute velocities from position changes."""
        if states.dim() == 3:  # [seq, objects, state_dim]
            start_states = states[0]
            end_states = states[-1]
            last_mask = mask[-1] if mask.dim() == 2 else mask
            seq_len = states.shape[0]
            computed_velocities = (end_states[:, self.POSITION_IDX] - start_states[:, self.POSITION_IDX]) / max(seq_len - 1, 1)
        else:
            end_states = states
            start_states = states
            last_mask = mask
            computed_velocities = states[:, self.VELOCITY_IDX]
        
        active_mask = last_mask.bool()
        count = int(active_mask.sum().item())
        
        return end_states, start_states, computed_velocities, active_mask, count
    
    def _answer_counting(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: How many objects?"""
        _, _, _, _, count = self._get_active_objects(states, mask)
        return str(count), {"count": count}
    
    def _answer_moving_count(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: How many objects are moving? Uses position change over time."""
        end_states, start_states, computed_vel, active_mask, _ = self._get_active_objects(states, mask)
        
        if states.dim() == 3:
            pos_change = (end_states[:, self.POSITION_IDX] - start_states[:, self.POSITION_IDX]).abs().sum(dim=-1)
            moving = (pos_change > self.position_change_threshold) & active_mask
        else:
            speeds = torch.norm(computed_vel, dim=-1)
            moving = (speeds > self.position_change_threshold) & active_mask
        
        count = int(moving.sum().item())
        return str(count), {"moving_count": count}
    
    def _answer_direction_count(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        direction: str
    ) -> Tuple[str, Dict]:
        """Answer: How many objects moving in direction? Uses position change."""
        end_states, start_states, computed_vel, active_mask, _ = self._get_active_objects(states, mask)
        
        if states.dim() == 3:
            pos_diff = end_states[:, self.POSITION_IDX] - start_states[:, self.POSITION_IDX]
            pos_change = pos_diff.abs().sum(dim=-1)
            is_moving = pos_change > self.position_change_threshold
            vel_threshold = 0.001
        else:
            pos_diff = computed_vel
            is_moving = torch.norm(computed_vel, dim=-1) > self.position_change_threshold
            vel_threshold = self.position_change_threshold
        
        if direction == 'right':
            moving = (pos_diff[:, 0] > vel_threshold) & is_moving & active_mask
        elif direction == 'left':
            moving = (pos_diff[:, 0] < -vel_threshold) & is_moving & active_mask
        elif direction == 'up':
            moving = (pos_diff[:, 1] > vel_threshold) & is_moving & active_mask
        elif direction == 'down':
            moving = (pos_diff[:, 1] < -vel_threshold) & is_moving & active_mask
        else:
            moving = torch.zeros_like(active_mask)
        
        count = int(moving.sum().item())
        return str(count), {"direction": direction, "count": count}
    
    def _answer_collision(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Will objects collide? Uses position changes to detect approaching objects."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "No", {"will_collide": False, "reason": "less than 2 objects"}
        
        positions = end_states[:, self.POSITION_IDX]
        
        active_indices = torch.where(active_mask)[0]
        
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx_i, idx_j = active_indices[i], active_indices[j]
                
                pos_diff = positions[idx_j] - positions[idx_i]
                vel_diff = computed_vel[idx_j] - computed_vel[idx_i]
                
                distance = torch.norm(pos_diff).item()
                closing_speed = -torch.dot(pos_diff, vel_diff).item() / (distance + 1e-6)
                
                pos_change_i = (end_states[idx_i, self.POSITION_IDX] - start_states[idx_i, self.POSITION_IDX]).abs().sum().item()
                pos_change_j = (end_states[idx_j, self.POSITION_IDX] - start_states[idx_j, self.POSITION_IDX]).abs().sum().item()
                either_moving = pos_change_i > self.position_change_threshold or pos_change_j > self.position_change_threshold
                
                if distance < self.collision_threshold and either_moving:
                    return "Yes", {
                        "will_collide": True,
                        "objects": (int(idx_i), int(idx_j)),
                        "distance": distance
                    }
        
        return "No", {"will_collide": False}
    
    def _answer_energy(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Total kinetic energy using computed velocities."""
        end_states, start_states, computed_vel, active_mask, _ = self._get_active_objects(states, mask)
        
        masses = end_states[:, self.MASS_IDX]
        masses = torch.where(masses > 0, masses, torch.ones_like(masses))
        
        speeds_sq = (computed_vel ** 2).sum(dim=-1)
        kinetic_energy = 0.5 * masses * speeds_sq
        
        total_ke = (kinetic_energy * active_mask.float()).sum().item()
        
        if total_ke < 0.001:
            answer = "very low"
        elif total_ke < 0.01:
            answer = "low"
        elif total_ke < 0.1:
            answer = "moderate"
        elif total_ke < 1.0:
            answer = "high"
        else:
            answer = "very high"
        
        return answer, {"kinetic_energy": total_ke}
    
    def _answer_momentum(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Total momentum magnitude using computed velocities."""
        end_states, start_states, computed_vel, active_mask, _ = self._get_active_objects(states, mask)
        
        masses = end_states[:, self.MASS_IDX].unsqueeze(-1)
        masses = torch.where(masses > 0, masses, torch.ones_like(masses))
        
        momentum = masses * computed_vel
        momentum_masked = momentum * active_mask.float().unsqueeze(-1)
        
        total_momentum = momentum_masked.sum(dim=0)
        magnitude = torch.norm(total_momentum).item()
        
        if magnitude < 0.01:
            answer = "very low"
        elif magnitude < 0.1:
            answer = "low"
        elif magnitude < 1.0:
            answer = "moderate"
        elif magnitude < 10.0:
            answer = "high"
        else:
            answer = "very high"
        
        return answer, {"momentum_magnitude": magnitude}
    
    def _answer_largest_position(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Where is the largest object?"""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count == 0:
            return "No objects", {"largest_idx": -1}
        
        radii = end_states[:, self.RADIUS_IDX]
        radii_masked = radii * active_mask.float()
        
        largest_idx = int(radii_masked.argmax().item())
        position = end_states[largest_idx, self.POSITION_IDX]
        
        x, y = position[0].item(), position[1].item()
        
        if x > 0.5:
            x_desc = "right"
        elif x < -0.5:
            x_desc = "left"
        else:
            x_desc = "center"
        
        if y > 0.5:
            y_desc = "top"
        elif y < -0.5:
            y_desc = "bottom"
        else:
            y_desc = "middle"
        
        answer = f"{y_desc}-{x_desc}"
        return answer, {"largest_idx": largest_idx, "position": (x, y)}
    
    def _answer_fastest_object(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Which object is moving fastest? Uses position change."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count == 0:
            return "No objects", {"fastest_idx": -1}
        
        if states.dim() == 3:
            pos_change = (end_states[:, self.POSITION_IDX] - start_states[:, self.POSITION_IDX]).abs().sum(dim=-1)
        else:
            pos_change = torch.norm(computed_vel, dim=-1)
        
        pos_change_masked = pos_change * active_mask.float()
        
        fastest_idx = int(pos_change_masked.argmax().item())
        max_change = pos_change_masked[fastest_idx].item()
        
        if max_change < self.position_change_threshold:
            return "None are moving", {"fastest_idx": -1, "speed": 0}
        
        return f"Object {fastest_idx + 1}", {"fastest_idx": fastest_idx, "speed": max_change}

    def _answer_kinetic_energy_calc(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Total kinetic energy as a categorical bucket."""
        end_states, start_states, computed_vel, active_mask, _ = self._get_active_objects(states, mask)
        
        masses = end_states[:, self.MASS_IDX]
        masses = torch.where(masses > 0, masses, torch.ones_like(masses))
        
        speeds_sq = (computed_vel ** 2).sum(dim=-1)
        kinetic_energy = 0.5 * masses * speeds_sq
        
        total_ke = (kinetic_energy * active_mask.float()).sum().item()
        
        if total_ke < 0.001:
            answer = "negligible"
        elif total_ke < 0.01:
            answer = "very low"
        elif total_ke < 0.1:
            answer = "low"
        elif total_ke < 1.0:
            answer = "moderate"
        elif total_ke < 10.0:
            answer = "high"
        else:
            answer = "very high"
        
        return answer, {"kinetic_energy": total_ke, "bucket": answer}

    def _answer_momentum_calc(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Total momentum magnitude as a categorical bucket."""
        end_states, start_states, computed_vel, active_mask, _ = self._get_active_objects(states, mask)
        
        masses = end_states[:, self.MASS_IDX].unsqueeze(-1)
        masses = torch.where(masses > 0, masses, torch.ones_like(masses))
        
        momentum = masses * computed_vel
        momentum_masked = momentum * active_mask.float().unsqueeze(-1)
        
        total_momentum = momentum_masked.sum(dim=0)
        magnitude = torch.norm(total_momentum).item()
        
        if magnitude < 0.001:
            answer = "negligible"
        elif magnitude < 0.01:
            answer = "very low"
        elif magnitude < 0.1:
            answer = "low"
        elif magnitude < 1.0:
            answer = "moderate"
        elif magnitude < 10.0:
            answer = "high"
        else:
            answer = "very high"
        
        return answer, {"momentum_magnitude": magnitude, "bucket": answer}

    def _answer_max_speed_calc(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Maximum speed in the scene as a categorical bucket."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count == 0:
            return "stationary", {"max_speed": 0}
        
        speeds = torch.norm(computed_vel, dim=-1)
        speeds_masked = speeds * active_mask.float()
        
        max_speed = speeds_masked.max().item()
        
        if max_speed < 0.001:
            answer = "stationary"
        elif max_speed < 0.01:
            answer = "very slow"
        elif max_speed < 0.05:
            answer = "slow"
        elif max_speed < 0.2:
            answer = "moderate"
        elif max_speed < 1.0:
            answer = "fast"
        else:
            answer = "very fast"
        
        return answer, {"max_speed": max_speed, "bucket": answer}

    def _answer_min_distance_calc(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Minimum distance between any two objects."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "N/A", {"min_distance": float('inf'), "reason": "less than 2 objects"}
        
        positions = end_states[:, self.POSITION_IDX]
        active_indices = torch.where(active_mask)[0]
        
        min_dist = float('inf')
        closest_pair = (-1, -1)
        
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx_i, idx_j = active_indices[i], active_indices[j]
                dist = torch.norm(positions[idx_j] - positions[idx_i]).item()
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (int(idx_i), int(idx_j))
        
        if min_dist < 0.1:
            answer = "touching"
        elif min_dist < 0.5:
            answer = "very close"
        elif min_dist < 2.0:
            answer = "close"
        elif min_dist < 5.0:
            answer = "moderate"
        elif min_dist < 10.0:
            answer = "far"
        else:
            answer = "very far"
        
        return answer, {"min_distance": min_dist, "closest_pair": closest_pair, "bucket": answer}

    def _answer_time_to_collision(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Estimated timesteps until collision for closest approaching pair."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "N/A", {"time_to_collision": float('inf'), "reason": "less than 2 objects"}
        
        positions = end_states[:, self.POSITION_IDX]
        active_indices = torch.where(active_mask)[0]
        
        min_time = float('inf')
        collision_pair = (-1, -1)
        
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx_i, idx_j = active_indices[i], active_indices[j]
                
                pos_diff = positions[idx_j] - positions[idx_i]
                vel_diff = computed_vel[idx_j] - computed_vel[idx_i]
                
                distance = torch.norm(pos_diff).item()
                closing_speed = -torch.dot(pos_diff, vel_diff).item() / (distance + 1e-6)
                
                if closing_speed > 0.001:
                    time_to_collision = distance / closing_speed
                    if time_to_collision < min_time:
                        min_time = time_to_collision
                        collision_pair = (int(idx_i), int(idx_j))
        
        if min_time == float('inf'):
            return "never", {"time_to_collision": float('inf'), "reason": "no approaching pairs"}
        elif min_time > 500:
            return "very long", {"time_to_collision": min_time}
        elif min_time > 100:
            return "long", {"time_to_collision": min_time, "collision_pair": collision_pair}
        elif min_time > 30:
            return "moderate", {"time_to_collision": min_time, "collision_pair": collision_pair}
        elif min_time > 10:
            return "soon", {"time_to_collision": min_time, "collision_pair": collision_pair}
        else:
            return "imminent", {"time_to_collision": min_time, "collision_pair": collision_pair}

    def _answer_relative_motion(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Whether the closest pair is approaching or moving apart."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "N/A", {"relative_motion": None, "reason": "less than 2 objects"}
        
        positions = end_states[:, self.POSITION_IDX]
        active_indices = torch.where(active_mask)[0]
        
        min_dist = float('inf')
        closest_i, closest_j = 0, 1
        
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx_i, idx_j = active_indices[i], active_indices[j]
                dist = torch.norm(positions[idx_j] - positions[idx_i]).item()
                if dist < min_dist:
                    min_dist = dist
                    closest_i, closest_j = idx_i, idx_j
        
        pos_diff = positions[closest_j] - positions[closest_i]
        vel_diff = computed_vel[closest_j] - computed_vel[closest_i]
        
        closing_speed = -torch.dot(pos_diff, vel_diff).item() / (min_dist + 1e-6)
        
        if closing_speed > 0.001:
            return "approaching", {"relative_motion": "approaching", "closing_speed": closing_speed}
        elif closing_speed < -0.001:
            return "moving apart", {"relative_motion": "moving apart", "closing_speed": closing_speed}
        else:
            return "stationary", {"relative_motion": "stationary", "closing_speed": closing_speed}

    def _answer_fastest_object_num(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Index of the fastest moving object."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count == 0:
            return "none", {"fastest_idx": -1}
        
        speeds = torch.norm(computed_vel, dim=-1)
        speeds_masked = speeds * active_mask.float()
        
        fastest_idx = int(speeds_masked.argmax().item())
        max_speed = speeds_masked[fastest_idx].item()
        
        if max_speed < 0.0001:
            return "none", {"fastest_idx": -1, "speed": 0}
        
        return str(fastest_idx + 1), {"fastest_idx": fastest_idx, "speed": max_speed}

    def _answer_heaviest_object_num(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Index of the heaviest object."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count == 0:
            return "none", {"heaviest_idx": -1}
        
        masses = end_states[:, self.MASS_IDX]
        masses_masked = masses * active_mask.float()
        
        heaviest_idx = int(masses_masked.argmax().item())
        max_mass = masses_masked[heaviest_idx].item()
        
        return str(heaviest_idx + 1), {"heaviest_idx": heaviest_idx, "mass": max_mass}

    def _answer_will_collide_soon(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Will object 0 (player) collide with anything in next 10 frames?"""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "no", {"will_collide": False, "reason": "not enough objects"}
        
        player_pos = end_states[0, self.POSITION_IDX]
        player_vel = computed_vel[0]
        player_radius = end_states[0, self.RADIUS_IDX].item() if end_states.shape[1] > 7 else 0.5
        
        frames_to_check = 10
        
        for frame in range(1, frames_to_check + 1):
            future_player_pos = player_pos + player_vel * frame
            
            for i in range(1, count):
                if not active_mask[i]:
                    continue
                obj_pos = end_states[i, self.POSITION_IDX]
                obj_vel = computed_vel[i]
                obj_radius = end_states[i, self.RADIUS_IDX].item() if end_states.shape[1] > 7 else 0.5
                
                future_obj_pos = obj_pos + obj_vel * frame
                distance = torch.norm(future_player_pos - future_obj_pos).item()
                collision_dist = player_radius + obj_radius
                
                if distance < collision_dist:
                    return "yes", {"will_collide": True, "frame": frame, "object": i + 1}
        
        return "no", {"will_collide": False, "frames_checked": frames_to_check}

    def _answer_is_grounded(self, states: torch.Tensor, mask: torch.Tensor, obj_idx: int) -> Tuple[str, Dict]:
        """Answer: Is the specified object on the ground (y near 0)?"""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if obj_idx >= count or not active_mask[obj_idx]:
            return "unknown", {"grounded": None, "reason": "object not found"}
        
        y_pos = end_states[obj_idx, 1].item()
        y_vel = computed_vel[obj_idx, 1].item()
        
        ground_threshold = 0.1
        is_grounded = y_pos < ground_threshold and abs(y_vel) < 0.01
        
        if is_grounded:
            return "yes", {"grounded": True, "y_position": y_pos, "y_velocity": y_vel}
        else:
            return "no", {"grounded": False, "y_position": y_pos, "y_velocity": y_vel}

    def _answer_threat_level(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Threat level based on nearby fast-moving objects."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "safe", {"threat_level": 0, "reason": "no other objects"}
        
        player_pos = end_states[0, self.POSITION_IDX]
        threat_score = 0.0
        threats = []
        
        for i in range(1, count):
            if not active_mask[i]:
                continue
            
            obj_pos = end_states[i, self.POSITION_IDX]
            obj_vel = computed_vel[i]
            distance = torch.norm(obj_pos - player_pos).item()
            speed = torch.norm(obj_vel).item()
            
            direction_to_player = player_pos - obj_pos
            if torch.norm(direction_to_player) > 0.01:
                direction_to_player = direction_to_player / torch.norm(direction_to_player)
                approach_speed = torch.dot(obj_vel, direction_to_player).item()
            else:
                approach_speed = 0
            
            if distance < 5.0 and approach_speed > 0:
                obj_threat = (approach_speed / (distance + 0.1)) * 10
                threat_score += obj_threat
                threats.append({"object": i + 1, "distance": distance, "approach_speed": approach_speed})
        
        if threat_score < 0.5:
            level = "safe"
        elif threat_score < 2.0:
            level = "low"
        elif threat_score < 5.0:
            level = "medium"
        elif threat_score < 10.0:
            level = "high"
        else:
            level = "critical"
        
        return level, {"threat_level": threat_score, "threats": threats}

    def _answer_nearest_object(self, states: torch.Tensor, mask: torch.Tensor, ref_idx: int) -> Tuple[str, Dict]:
        """Answer: Which object is nearest to the reference object?"""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "none", {"nearest_idx": -1, "reason": "not enough objects"}
        
        if ref_idx >= count or not active_mask[ref_idx]:
            return "unknown", {"nearest_idx": -1, "reason": "reference object not found"}
        
        ref_pos = end_states[ref_idx, self.POSITION_IDX]
        min_dist = float('inf')
        nearest_idx = -1
        
        for i in range(count):
            if i == ref_idx or not active_mask[i]:
                continue
            
            obj_pos = end_states[i, self.POSITION_IDX]
            dist = torch.norm(obj_pos - ref_pos).item()
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        if nearest_idx == -1:
            return "none", {"nearest_idx": -1}
        
        return f"Object {nearest_idx + 1}", {"nearest_idx": nearest_idx, "distance": min_dist}

    def _answer_objects_in_range(self, states: torch.Tensor, mask: torch.Tensor, ref_idx: int, range_dist: float) -> Tuple[str, Dict]:
        """Answer: How many objects are within range of reference object?"""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if ref_idx >= count or not active_mask[ref_idx]:
            return "0", {"count": 0, "reason": "reference object not found"}
        
        ref_pos = end_states[ref_idx, self.POSITION_IDX]
        in_range = []
        
        for i in range(count):
            if i == ref_idx or not active_mask[i]:
                continue
            
            obj_pos = end_states[i, self.POSITION_IDX]
            dist = torch.norm(obj_pos - ref_pos).item()
            
            if dist <= range_dist:
                in_range.append({"object": i + 1, "distance": dist})
        
        count_in_range = len(in_range)
        if count_in_range == 0:
            answer = "0"
        elif count_in_range == 1:
            answer = "1"
        elif count_in_range <= 3:
            answer = "2-3"
        else:
            answer = "4+"

        return answer, {
            "objects_in_range_count": count_in_range,
            "objects_in_range": in_range,
            "range": range_dist,
            "bucket": answer,
        }

    def _answer_trajectory_prediction(self, states: torch.Tensor, mask: torch.Tensor, obj_idx: int, frames: int) -> Tuple[str, Dict]:
        """Answer: Predict where object will be in N frames (linear extrapolation)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if obj_idx >= count or not active_mask[obj_idx]:
            return "unknown", {"predicted_pos": None, "reason": "object not found"}
        
        current_pos = end_states[obj_idx, self.POSITION_IDX]
        velocity = computed_vel[obj_idx]
        
        predicted_pos = current_pos + velocity * frames
        x, y, z = predicted_pos[0].item(), predicted_pos[1].item(), predicted_pos[2].item()
        
        if abs(x) < 0.5 and abs(y) < 0.5 and abs(z) < 0.5:
            answer = "center"
        elif x > 2:
            answer = "far right"
        elif x > 0.5:
            answer = "right"
        elif x < -2:
            answer = "far left"
        elif x < -0.5:
            answer = "left"
        elif y > 1:
            answer = "high"
        elif y < -1:
            answer = "low"
        else:
            answer = "center"
        
        return answer, {"predicted_pos": [x, y, z], "frames": frames, "velocity": velocity.tolist()}

    def _answer_can_reach_target(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Can object 0 reach a target position before object 1?"""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "unknown", {"can_reach": None, "reason": "not enough objects"}
        
        target_pos = torch.tensor([5.0, 0.0, 0.0], device=end_states.device)
        
        pos_0 = end_states[0, self.POSITION_IDX]
        vel_0 = computed_vel[0]
        dist_0 = torch.norm(target_pos - pos_0).item()
        speed_0 = torch.norm(vel_0).item()
        
        pos_1 = end_states[1, self.POSITION_IDX]
        vel_1 = computed_vel[1]
        dist_1 = torch.norm(target_pos - pos_1).item()
        speed_1 = torch.norm(vel_1).item()
        
        time_0 = dist_0 / (speed_0 + 0.001)
        time_1 = dist_1 / (speed_1 + 0.001)
        
        if time_0 < time_1:
            return "yes", {"can_reach": True, "time_obj0": time_0, "time_obj1": time_1}
        else:
            return "no", {"can_reach": False, "time_obj0": time_0, "time_obj1": time_1}

    def _answer_is_path_blocked(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Is the path from object 0 to object 2 blocked by object 1?"""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 3:
            return "no", {"blocked": False, "reason": "not enough objects"}
        
        start_pos = end_states[0, self.POSITION_IDX]
        end_pos = end_states[2, self.POSITION_IDX]
        blocker_pos = end_states[1, self.POSITION_IDX]
        blocker_radius = end_states[1, self.RADIUS_IDX].item() if end_states.shape[1] > 7 else 0.5
        
        path_vec = end_pos - start_pos
        path_length = torch.norm(path_vec).item()
        
        if path_length < 0.01:
            return "no", {"blocked": False, "reason": "start and end too close"}
        
        path_dir = path_vec / path_length
        to_blocker = blocker_pos - start_pos
        
        projection = torch.dot(to_blocker, path_dir).item()
        
        if projection < 0 or projection > path_length:
            return "no", {"blocked": False, "reason": "blocker not on path"}
        
        closest_point = start_pos + path_dir * projection
        distance_to_path = torch.norm(blocker_pos - closest_point).item()
        
        if distance_to_path < blocker_radius + 0.5:
            return "yes", {"blocked": True, "distance_to_path": distance_to_path}
        else:
            return "no", {"blocked": False, "distance_to_path": distance_to_path}

    def _answer_metaphor_collision(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map collision physics to social/conceptual metaphors."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "no interaction - only one entity present", {"collision_type": "none", "count": count}
        
        pos_0 = end_states[0, self.POSITION_IDX]
        pos_1 = end_states[1, self.POSITION_IDX]
        vel_0 = computed_vel[0]
        vel_1 = computed_vel[1]
        
        speed_0 = torch.norm(vel_0).item()
        speed_1 = torch.norm(vel_1).item()
        mass_0 = end_states[0, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
        mass_1 = end_states[1, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
        
        momentum_0 = speed_0 * mass_0
        momentum_1 = speed_1 * mass_1
        
        rel_vel = torch.norm(vel_0 - vel_1).item()
        distance = torch.norm(pos_0 - pos_1).item()
        
        if rel_vel < 0.002:
            answer = "Both parties are at a standstill - no clash of ideas, perhaps agreement or stalemate"
            collision_type = "static"
        elif momentum_0 > momentum_1 * 2:
            answer = "The first argument dominates - it has much more force and will likely overwhelm the counterargument"
            collision_type = "dominant_first"
        elif momentum_1 > momentum_0 * 2:
            answer = "The counterargument is stronger - it carries more weight and will likely prevail"
            collision_type = "dominant_second"
        elif rel_vel > 0.02:
            answer = "A heated clash - both ideas collide forcefully, likely causing significant change in both positions"
            collision_type = "intense"
        else:
            answer = "A moderate exchange - both arguments meet with similar force, leading to mutual adjustment"
            collision_type = "balanced"
        
        return answer, {
            "collision_type": collision_type,
            "momentum_ratio": momentum_0 / (momentum_1 + 0.001),
            "relative_velocity": rel_vel,
            "distance": distance
        }

    def _answer_metaphor_momentum(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map momentum to career/emotional/political progress metaphors."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no entity to analyze", {"momentum_level": "none"}
        
        all_speeds = []
        all_momenta = []
        for i in range(count):
            if active_mask[i]:
                vel = computed_vel[i]
                speed = torch.norm(vel).item()
                mass = end_states[i, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
                all_speeds.append(speed)
                all_momenta.append(speed * mass)
        
        if not all_momenta:
            return "no entity to analyze", {"momentum_level": "none"}
        
        max_momentum = max(all_momenta)
        max_speed = max(all_speeds)
        avg_momentum = sum(all_momenta) / len(all_momenta)
        
        vel = computed_vel[0]
        vel_y = vel[1].item() if len(vel) > 1 else 0
        vel_z = vel[2].item() if len(vel) > 2 else 0
        vertical = vel_y + vel_z
        
        momentum_answers = [
            ("Stagnant - no forward progress, stuck in place with no momentum to speak of", "stagnant"),
            ("Slowly rising - gradual upward progress, building momentum carefully", "rising_slow"),
            ("Declining slowly - losing ground but not in freefall, still recoverable", "declining_slow"),
            ("Coasting - moving forward but without strong drive, maintaining status quo", "coasting"),
            ("Powerful forward drive - strong horizontal progress, making significant headway", "driving_forward"),
            ("Meteoric rise - unstoppable upward momentum, skyrocketing success", "meteoric_rise"),
            ("Crashing down - overwhelming negative momentum, a dramatic fall", "crashing"),
            ("Unstoppable force - massive momentum carrying them forward relentlessly", "unstoppable"),
        ]
        
        if max_speed < 1e-8:
            weights = [0.7, 0.05, 0.05, 0.1, 0.05, 0.02, 0.02, 0.01]
        elif vertical > max_speed * 0.3:
            weights = [0.05, 0.35, 0.05, 0.15, 0.1, 0.2, 0.02, 0.08]
        elif vertical < -max_speed * 0.3:
            weights = [0.05, 0.05, 0.35, 0.15, 0.1, 0.02, 0.2, 0.08]
        elif max_momentum > avg_momentum * 1.5:
            weights = [0.02, 0.1, 0.1, 0.15, 0.25, 0.13, 0.1, 0.15]
        else:
            weights = [0.1, 0.15, 0.15, 0.25, 0.2, 0.05, 0.05, 0.05]
        
        idx = self.rng.choices(range(len(momentum_answers)), weights=weights, k=1)[0]
        answer, level = momentum_answers[idx]
        
        return answer, {
            "momentum_level": level,
            "momentum_magnitude": max_momentum,
            "vertical_component": vertical,
            "speed": max_speed
        }

    def _answer_metaphor_equilibrium(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map system equilibrium to relationship/work-life balance metaphors."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "single entity - no balance to assess", {"equilibrium_type": "single"}
        
        total_momentum = torch.zeros(3)
        total_ke = 0.0
        center_of_mass = torch.zeros(3)
        total_mass = 0.0
        
        for i in range(count):
            if not active_mask[i]:
                continue
            vel = computed_vel[i]
            mass = end_states[i, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
            pos = end_states[i, self.POSITION_IDX]
            
            total_momentum += vel * mass
            total_ke += 0.5 * mass * torch.norm(vel).item() ** 2
            center_of_mass += pos * mass
            total_mass += mass
        
        net_momentum = torch.norm(total_momentum).item()
        avg_ke = total_ke / count
        
        spread = 0.0
        for i in range(count):
            if active_mask[i]:
                pos = end_states[i, self.POSITION_IDX]
                spread += torch.norm(pos - center_of_mass / total_mass).item()
        spread /= count
        
        individual_momenta = []
        for i in range(count):
            if active_mask[i]:
                vel = computed_vel[i]
                mass = end_states[i, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
                individual_momenta.append(torch.norm(vel).item() * mass)
        
        total_individual = sum(individual_momenta) if individual_momenta else 1e-10
        momentum_cancellation = 1.0 - (net_momentum / (total_individual + 1e-10))
        ke_variance = max(individual_momenta) / (avg_ke + 1e-10) if individual_momenta else 0
        
        equilibrium_answers = [
            ("Perfect equilibrium - all forces balanced, a stable and harmonious state like a well-balanced relationship", "stable"),
            ("Near balance - minor tensions exist but overall stable, like a relationship with small disagreements", "near_stable"),
            ("Unbalanced - strong net force in one direction, like a relationship where one person dominates", "unbalanced"),
            ("Chaotic energy - high activity but no clear direction, like a turbulent work environment", "chaotic"),
            ("Dynamic equilibrium - active but balanced, like a busy but manageable work-life situation", "dynamic"),
        ]
        
        if momentum_cancellation > 0.9:
            weights = [0.4, 0.3, 0.05, 0.05, 0.2]
        elif momentum_cancellation > 0.6:
            weights = [0.15, 0.35, 0.1, 0.1, 0.3]
        elif momentum_cancellation < 0.4:
            weights = [0.05, 0.1, 0.5, 0.2, 0.15]
        elif spread > 3.0:
            weights = [0.05, 0.1, 0.15, 0.5, 0.2]
        else:
            weights = [0.1, 0.2, 0.2, 0.2, 0.3]
        
        idx = self.rng.choices(range(len(equilibrium_answers)), weights=weights, k=1)[0]
        answer, eq_type = equilibrium_answers[idx]
        
        return answer, {
            "equilibrium_type": eq_type,
            "net_momentum": net_momentum,
            "average_kinetic_energy": avg_ke,
            "spatial_spread": spread
        }

    def _answer_metaphor_trajectory(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map trajectory to life path/growth metaphors."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no entity to track", {"trajectory_type": "none"}
        
        pos = end_states[0, self.POSITION_IDX]
        start_pos = start_states[0, self.POSITION_IDX]
        displacement = pos - start_pos
        
        vel = computed_vel[0]
        speed = torch.norm(vel).item()
        
        disp_x = displacement[0].item()
        disp_y = displacement[1].item() if len(displacement) > 1 else 0
        disp_z = displacement[2].item() if len(displacement) > 2 else 0
        vertical_disp = disp_y + disp_z
        horizontal_disp = abs(disp_x)
        total_disp = torch.norm(displacement).item()
        
        trajectory_answers = [
            ("At a crossroads - stationary, perhaps contemplating the next move or stuck in indecision", "stationary"),
            ("Rapid ascent - shooting upward like a startup in hypergrowth or a student excelling beyond expectations", "rapid_ascent"),
            ("Steady climb - consistent upward progress, like a career with regular promotions", "steady_climb"),
            ("Sharp decline - falling fast, like a company losing market share rapidly", "sharp_decline"),
            ("Gradual descent - slowly losing altitude, perhaps a career plateau turning into decline", "gradual_descent"),
            ("Forward progress - moving ahead steadily, making horizontal gains like expanding reach", "forward_progress"),
            ("Retreating - moving backward, perhaps reconsidering past decisions or losing ground", "retreating"),
            ("Lateral movement - changing direction without clear up or down, exploring new paths", "lateral"),
        ]
        
        if total_disp < 1e-6:
            weights = [0.6, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]
        elif vertical_disp > horizontal_disp * 0.3 and vertical_disp > 0:
            weights = [0.05, 0.35, 0.3, 0.02, 0.03, 0.15, 0.02, 0.08]
        elif vertical_disp < -horizontal_disp * 0.3:
            weights = [0.05, 0.02, 0.03, 0.35, 0.3, 0.05, 0.12, 0.08]
        elif horizontal_disp > abs(vertical_disp) and disp_x > 0:
            weights = [0.05, 0.1, 0.15, 0.02, 0.03, 0.45, 0.05, 0.15]
        elif horizontal_disp > abs(vertical_disp) and disp_x < 0:
            weights = [0.05, 0.02, 0.03, 0.1, 0.15, 0.05, 0.45, 0.15]
        else:
            weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2]
        
        idx = self.rng.choices(range(len(trajectory_answers)), weights=weights, k=1)[0]
        answer, traj_type = trajectory_answers[idx]
        
        return answer, {
            "trajectory_type": traj_type,
            "speed": speed,
            "vertical_displacement": vertical_disp,
            "horizontal_displacement": horizontal_disp,
            "total_displacement": total_disp
        }

    def _answer_metaphor_force(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map forces to social pressure/influence metaphors."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no entity experiencing forces", {"force_type": "none"}
        
        if count == 1:
            vel = computed_vel[0]
            accel_approx = vel
            force_mag = torch.norm(accel_approx).item()
            
            if force_mag < 0.001:
                answer = "No external pressure - free to move as desired, no competing influences"
                force_type = "free"
            else:
                answer = "Internal drive - moving under own motivation without external pressures"
                force_type = "self_driven"
            
            return answer, {"force_type": force_type, "force_magnitude": force_mag}
        
        ref_pos = end_states[0, self.POSITION_IDX]
        net_force = torch.zeros(3)
        
        for i in range(1, count):
            if not active_mask[i]:
                continue
            other_pos = end_states[i, self.POSITION_IDX]
            direction = other_pos - ref_pos
            dist = torch.norm(direction).item()
            if dist > 0.01:
                other_mass = end_states[i, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
                force_strength = other_mass / (dist ** 2 + 0.1)
                net_force += direction / dist * force_strength
        
        force_mag = torch.norm(net_force).item()
        force_y = net_force[1].item() if len(net_force) > 1 else 0
        
        force_answers = [
            ("Balanced pressures - competing influences cancel out, leaving freedom of choice", "balanced"),
            ("Gentle upward pressure - mild encouragement toward growth or improvement", "gentle_up"),
            ("Subtle downward pressure - slight discouragement or resistance to progress", "gentle_down"),
            ("Mild lateral pressure - gentle push toward a new direction", "mild_lateral"),
            ("Strong uplifting influence - powerful forces pushing toward success and growth", "strong_up"),
            ("Heavy suppressive pressure - strong forces pushing down, like overwhelming criticism or obstacles", "strong_down"),
            ("Powerful lateral forces - strong pressure to change course dramatically", "strong_lateral"),
        ]
        
        force_ratio = abs(force_y) / (force_mag + 1e-10)
        
        if force_mag < 0.1:
            weights = [0.5, 0.1, 0.1, 0.15, 0.05, 0.05, 0.05]
        elif force_ratio > 0.5 and force_y > 0:
            weights = [0.1, 0.35, 0.05, 0.1, 0.3, 0.02, 0.08]
        elif force_ratio > 0.5 and force_y < 0:
            weights = [0.1, 0.05, 0.35, 0.1, 0.02, 0.3, 0.08]
        elif force_mag > 1.0:
            weights = [0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.2]
        else:
            weights = [0.2, 0.15, 0.15, 0.2, 0.1, 0.1, 0.1]
        
        idx = self.rng.choices(range(len(force_answers)), weights=weights, k=1)[0]
        answer, force_type = force_answers[idx]
        
        return answer, {
            "force_type": force_type,
            "force_magnitude": force_mag,
            "vertical_force": force_y,
            "num_influencers": count - 1
        }

    def _answer_metaphor_container(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map spatial containment to conceptual containment (Lakoff's CONTAINER schema)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "empty container - no ideas present", {"container_type": "empty", "count": 0}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        center = positions.mean(dim=0)
        
        distances_from_center = torch.norm(positions - center, dim=1)
        avg_dist = distances_from_center.mean().item()
        max_dist = distances_from_center.max().item()
        
        inside_count = (distances_from_center < avg_dist).sum().item()
        outside_count = count - inside_count
        
        boundary_size = max_dist * 1.2
        
        container_answers = [
            ("Tightly contained - all ideas are closely held within clear boundaries, a focused mindset", "tight"),
            ("Loosely contained - ideas spread across the space but still within bounds, flexible thinking", "loose"),
            ("Overflowing - some ideas escape the container, thoughts spilling beyond intended scope", "overflow"),
            ("Sparse interior - few ideas inside, mostly empty space, room for more", "sparse"),
            ("Densely packed - many ideas crammed together, possibly overwhelming", "dense"),
            ("Centered core - key ideas at the center with supporting thoughts around the periphery", "centered"),
        ]
        
        density = count / (boundary_size ** 2 + 0.1)
        spread_ratio = avg_dist / (max_dist + 0.01)
        
        if density > 0.5:
            weights = [0.1, 0.1, 0.15, 0.05, 0.4, 0.2]
        elif spread_ratio > 0.7:
            weights = [0.35, 0.25, 0.05, 0.1, 0.1, 0.15]
        elif spread_ratio < 0.4:
            weights = [0.1, 0.15, 0.1, 0.15, 0.1, 0.4]
        elif outside_count > inside_count:
            weights = [0.05, 0.2, 0.4, 0.1, 0.1, 0.15]
        else:
            weights = [0.2, 0.3, 0.1, 0.15, 0.1, 0.15]
        
        idx = self.rng.choices(range(len(container_answers)), weights=weights, k=1)[0]
        answer, container_type = container_answers[idx]
        
        return answer, {
            "container_type": container_type,
            "inside_count": inside_count,
            "outside_count": outside_count,
            "boundary_size": boundary_size,
            "density": density
        }

    def _answer_metaphor_source_path_goal(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map motion to journey metaphor (Lakoff's SOURCE-PATH-GOAL schema)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no traveler on this journey", {"journey_type": "none"}
        
        start_pos = start_states[0, self.POSITION_IDX]
        end_pos = end_states[0, self.POSITION_IDX]
        vel = computed_vel[0]
        
        displacement = end_pos - start_pos
        distance_traveled = torch.norm(displacement).item()
        speed = torch.norm(vel).item()
        
        if speed > 0.001:
            projected_goal = end_pos + vel * 10
            goal_distance = torch.norm(projected_goal - start_pos).item()
        else:
            projected_goal = end_pos
            goal_distance = distance_traveled
        
        progress_ratio = distance_traveled / (goal_distance + 0.01)
        
        journey_answers = [
            ("Just beginning - at the starting point, the journey ahead is long but full of possibility", "beginning"),
            ("Early stages - made initial progress, still finding footing on the path", "early"),
            ("Midway through - significant progress made, can see both where you came from and where you're going", "midway"),
            ("Approaching the goal - the destination is in sight, final push needed", "approaching"),
            ("Arrived - reached the destination, journey complete", "arrived"),
            ("Lost direction - movement without clear progress toward any goal", "lost"),
            ("Detour - veered off the main path, exploring alternative routes", "detour"),
            ("Stalled - stopped on the path, obstacles or decisions blocking progress", "stalled"),
        ]
        
        if speed < 0.001:
            weights = [0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.5]
        elif progress_ratio < 0.2:
            weights = [0.4, 0.3, 0.1, 0.02, 0.02, 0.08, 0.05, 0.03]
        elif progress_ratio < 0.5:
            weights = [0.1, 0.35, 0.3, 0.05, 0.02, 0.08, 0.07, 0.03]
        elif progress_ratio < 0.8:
            weights = [0.02, 0.1, 0.4, 0.3, 0.05, 0.05, 0.05, 0.03]
        elif progress_ratio < 1.0:
            weights = [0.02, 0.05, 0.15, 0.45, 0.2, 0.03, 0.05, 0.05]
        else:
            weights = [0.02, 0.02, 0.05, 0.1, 0.5, 0.1, 0.15, 0.06]
        
        idx = self.rng.choices(range(len(journey_answers)), weights=weights, k=1)[0]
        answer, journey_type = journey_answers[idx]
        
        return answer, {
            "journey_type": journey_type,
            "distance_traveled": distance_traveled,
            "progress_ratio": progress_ratio,
            "speed": speed
        }

    def _answer_metaphor_balance(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map mass/weight distribution to fairness/justice (Lakoff's BALANCE schema)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "single entity - no balance to assess", {"balance_type": "single"}
        
        masses = []
        positions = []
        for i in range(count):
            if active_mask[i]:
                mass = end_states[i, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
                pos = end_states[i, self.POSITION_IDX]
                masses.append(mass)
                positions.append(pos)
        
        total_mass = sum(masses)
        mass_ratios = [m / total_mass for m in masses]
        max_ratio = max(mass_ratios)
        min_ratio = min(mass_ratios)
        imbalance = max_ratio - min_ratio
        
        center_of_mass = sum(m * p for m, p in zip(masses, positions)) / total_mass
        geometric_center = sum(positions) / len(positions)
        com_offset = torch.norm(center_of_mass - geometric_center).item()
        
        balance_answers = [
            ("Perfect justice - responsibilities distributed equally, everyone carries fair share", "equal"),
            ("Slight imbalance - minor inequity but generally fair distribution", "slight_imbalance"),
            ("Moderate imbalance - noticeable disparity, some carry more than their share", "moderate_imbalance"),
            ("Severe imbalance - one party bears disproportionate burden, unfair distribution", "severe_imbalance"),
            ("Extreme concentration - almost all weight on one entity, others barely contributing", "extreme"),
            ("Spatially skewed - weight concentrated on one side, creating instability", "skewed"),
        ]
        
        if imbalance < 0.1:
            weights = [0.5, 0.3, 0.1, 0.05, 0.02, 0.03]
        elif imbalance < 0.3:
            weights = [0.15, 0.4, 0.25, 0.1, 0.05, 0.05]
        elif imbalance < 0.5:
            weights = [0.05, 0.15, 0.4, 0.25, 0.1, 0.05]
        elif imbalance < 0.7:
            weights = [0.02, 0.05, 0.2, 0.4, 0.25, 0.08]
        else:
            weights = [0.01, 0.02, 0.07, 0.2, 0.5, 0.2]
        
        if com_offset > 2.0:
            weights[-1] += 0.2
            weights = [w / sum(weights) for w in weights]
        
        idx = self.rng.choices(range(len(balance_answers)), weights=weights, k=1)[0]
        answer, balance_type = balance_answers[idx]
        
        return answer, {
            "balance_type": balance_type,
            "imbalance_ratio": imbalance,
            "max_burden": max(masses),
            "min_burden": min(masses),
            "com_offset": com_offset
        }

    def _answer_metaphor_link(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map spatial proximity to social relationships (Lakoff's LINK schema)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "isolated entity - no relationships to map", {"link_type": "isolated"}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        
        distances = []
        pairs = []
        for i in range(count):
            for j in range(i + 1, count):
                dist = torch.norm(positions[i] - positions[j]).item()
                distances.append(dist)
                pairs.append((i + 1, j + 1, dist))
        
        avg_dist = sum(distances) / len(distances)
        min_dist = min(distances)
        max_dist = max(distances)
        
        closest_pair = min(pairs, key=lambda x: x[2])
        farthest_pair = max(pairs, key=lambda x: x[2])
        
        link_answers = [
            (f"Close-knit network - all entities tightly connected, high trust throughout (closest: {closest_pair[0]}-{closest_pair[1]})", "tight"),
            (f"Strong bond between {closest_pair[0]} and {closest_pair[1]} - these two share deep trust, others more distant", "strong_pair"),
            (f"Loose network - connections exist but with distance, casual relationships", "loose"),
            (f"Fragmented - {farthest_pair[0]} and {farthest_pair[1]} are disconnected, trust broken between them", "fragmented"),
            ("Hub and spokes - one central connector with others linked through them", "hub"),
            ("Evenly distributed - moderate connections throughout, professional relationships", "distributed"),
        ]
        
        dist_variance = max_dist - min_dist
        
        if avg_dist < 1.5:
            weights = [0.4, 0.25, 0.1, 0.05, 0.1, 0.1]
        elif dist_variance > avg_dist:
            weights = [0.1, 0.3, 0.1, 0.3, 0.1, 0.1]
        elif avg_dist > 4.0:
            weights = [0.05, 0.1, 0.3, 0.35, 0.05, 0.15]
        else:
            weights = [0.15, 0.2, 0.25, 0.1, 0.1, 0.2]
        
        idx = self.rng.choices(range(len(link_answers)), weights=weights, k=1)[0]
        answer, link_type = link_answers[idx]
        
        return answer, {
            "link_type": link_type,
            "avg_distance": avg_dist,
            "closest_pair": closest_pair,
            "farthest_pair": farthest_pair,
            "num_connections": len(distances)
        }

    def _answer_metaphor_center_periphery(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map spatial position to importance/power (Lakoff's CENTER-PERIPHERY schema)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "single entity - it holds all the power by default", {"hierarchy_type": "single"}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        masses = []
        for i in range(count):
            if active_mask[i]:
                mass = end_states[i, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
                masses.append(mass)
        
        geometric_center = positions.mean(dim=0)
        distances_from_center = torch.norm(positions - geometric_center, dim=1)
        
        centrality_scores = []
        for i in range(count):
            dist = distances_from_center[i].item()
            mass = masses[i]
            centrality = mass / (dist + 0.5)
            centrality_scores.append((i + 1, centrality, dist, mass))
        
        centrality_scores.sort(key=lambda x: x[1], reverse=True)
        most_central = centrality_scores[0]
        most_peripheral = centrality_scores[-1]
        
        hierarchy_answers = [
            (f"Object {most_central[0]} dominates the center - it holds the most power and influence", "dominant_center"),
            (f"Object {most_peripheral[0]} is marginalized - pushed to the periphery with little influence", "marginalized"),
            ("Distributed power - no single entity dominates, influence is shared", "distributed"),
            ("Contested center - multiple entities vie for central position", "contested"),
            (f"Clear hierarchy - {most_central[0]} at top, {most_peripheral[0]} at bottom", "hierarchical"),
            ("Flat structure - all entities roughly equidistant from center, equal standing", "flat"),
        ]
        
        centrality_variance = max(c[1] for c in centrality_scores) - min(c[1] for c in centrality_scores)
        avg_dist = distances_from_center.mean().item()
        
        if centrality_variance > 2.0:
            weights = [0.35, 0.25, 0.05, 0.1, 0.2, 0.05]
        elif centrality_variance < 0.5:
            weights = [0.1, 0.1, 0.3, 0.1, 0.1, 0.3]
        elif most_central[2] < avg_dist * 0.5:
            weights = [0.4, 0.15, 0.1, 0.15, 0.15, 0.05]
        else:
            weights = [0.15, 0.15, 0.2, 0.2, 0.15, 0.15]
        
        idx = self.rng.choices(range(len(hierarchy_answers)), weights=weights, k=1)[0]
        answer, hierarchy_type = hierarchy_answers[idx]
        
        return answer, {
            "hierarchy_type": hierarchy_type,
            "most_central": most_central[0],
            "most_peripheral": most_peripheral[0],
            "centrality_variance": centrality_variance
        }

    def _answer_metaphor_resistance(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Map motion impediments to obstacles/challenges (Lakoff's BLOCKAGE schema)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "clear path - no obstacles present", {"resistance_type": "clear"}
        
        speeds = []
        accelerations = []
        for i in range(count):
            if active_mask[i]:
                vel = computed_vel[i]
                speed = torch.norm(vel).item()
                speeds.append(speed)
                
                start_vel = start_states[i, self.POSITION_IDX] - start_states[max(0, i-1), self.POSITION_IDX]
                accel = torch.norm(vel - start_vel).item()
                accelerations.append(accel)
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        min_speed = min(speeds) if speeds else 0
        speed_variance = max_speed - min_speed
        
        slow_count = sum(1 for s in speeds if s < avg_speed * 0.5)
        fast_count = sum(1 for s in speeds if s > avg_speed * 1.5)
        
        resistance_answers = [
            ("Smooth sailing - no significant resistance, progress flows freely", "none"),
            ("Minor friction - small obstacles slow progress but don't stop it", "minor"),
            ("Moderate resistance - noticeable barriers requiring effort to overcome", "moderate"),
            ("Heavy resistance - significant obstacles blocking progress, struggle evident", "heavy"),
            ("Gridlock - movement nearly impossible, overwhelming barriers", "gridlock"),
            ("Uneven resistance - some paths clear while others blocked, creating bottlenecks", "uneven"),
            ("Breakthrough moment - overcoming resistance, acceleration visible", "breakthrough"),
        ]
        
        if avg_speed < 0.01:
            weights = [0.05, 0.05, 0.1, 0.2, 0.5, 0.05, 0.05]
        elif speed_variance > avg_speed:
            weights = [0.1, 0.1, 0.15, 0.15, 0.1, 0.35, 0.05]
        elif avg_speed > 0.1:
            weights = [0.4, 0.25, 0.15, 0.05, 0.02, 0.08, 0.05]
        elif slow_count > fast_count:
            weights = [0.05, 0.15, 0.3, 0.3, 0.1, 0.05, 0.05]
        else:
            weights = [0.2, 0.25, 0.2, 0.1, 0.05, 0.1, 0.1]
        
        if max(accelerations) > avg_speed * 0.5 if avg_speed > 0 else False:
            weights[-1] += 0.15
            weights = [w / sum(weights) for w in weights]
        
        idx = self.rng.choices(range(len(resistance_answers)), weights=weights, k=1)[0]
        answer, resistance_type = resistance_answers[idx]
        
        return answer, {
            "resistance_type": resistance_type,
            "avg_speed": avg_speed,
            "speed_variance": speed_variance,
            "slow_count": slow_count,
            "fast_count": fast_count
        }

    # =========================================================================
    # Mathematical Metaphors from "Where Mathematics Comes From" (Lakoff & Núñez)
    # =========================================================================

    def _answer_metaphor_arithmetic_motion(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Arithmetic is motion along a path (number line walking)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no motion to map to arithmetic", {"arithmetic_type": "none"}
        
        start_pos = start_states[0, self.POSITION_IDX]
        end_pos = end_states[0, self.POSITION_IDX]
        displacement = end_pos - start_pos
        
        dx = displacement[0].item()
        total_displacement = torch.norm(displacement).item()
        
        steps_forward = max(0, int(dx * 10))
        steps_backward = max(0, int(-dx * 10))
        
        arithmetic_answers = [
            (f"Adding {steps_forward} - walking {steps_forward} steps forward along the number line", "addition"),
            (f"Subtracting {steps_backward} - walking {steps_backward} steps backward along the number line", "subtraction"),
            ("Zero operation - standing still at the current number", "identity"),
            (f"Net change of {int(dx * 10):+d} - the arithmetic sum of forward and backward steps", "net_change"),
            ("Approaching the origin - moving toward zero on the number line", "toward_zero"),
            ("Moving away from origin - increasing magnitude on the number line", "away_from_zero"),
        ]
        
        if abs(dx) < 0.01:
            weights = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]
        elif dx > 0.1:
            weights = [0.4, 0.05, 0.05, 0.25, 0.1, 0.15]
        elif dx < -0.1:
            weights = [0.05, 0.4, 0.05, 0.25, 0.15, 0.1]
        else:
            weights = [0.2, 0.2, 0.15, 0.25, 0.1, 0.1]
        
        idx = self.rng.choices(range(len(arithmetic_answers)), weights=weights, k=1)[0]
        answer, arith_type = arithmetic_answers[idx]
        
        return answer, {
            "arithmetic_type": arith_type,
            "displacement_x": dx,
            "total_displacement": total_displacement,
            "steps_forward": steps_forward,
            "steps_backward": steps_backward
        }

    def _answer_metaphor_arithmetic_collection(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Arithmetic is object collection (putting things together/apart)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "empty collection - nothing to count", {"collection_type": "empty", "count": 0}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        center = positions.mean(dim=0)
        
        distances = torch.norm(positions - center, dim=1)
        close_threshold = distances.mean().item()
        
        groups = []
        assigned = [False] * count
        for i in range(count):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, count):
                if not assigned[j]:
                    dist = torch.norm(positions[i] - positions[j]).item()
                    if dist < close_threshold:
                        group.append(j)
                        assigned[j] = True
            groups.append(group)
        
        num_groups = len(groups)
        group_sizes = [len(g) for g in groups]
        
        collection_answers = [
            (f"Collection of {count} - all objects gathered as a single sum", "single_collection"),
            (f"{num_groups} groups combining to {count} - addition: {' + '.join(map(str, group_sizes))} = {count}", "grouped_addition"),
            (f"Scattered collection - {count} individual units not yet combined", "scattered"),
            (f"Partitioned into {num_groups} subsets - the whole {count} divided into parts", "partition"),
            (f"Union of groups - combining {num_groups} collections yields {count} total", "union"),
        ]
        
        if num_groups == 1:
            weights = [0.5, 0.1, 0.2, 0.1, 0.1]
        elif num_groups == count:
            weights = [0.1, 0.1, 0.5, 0.2, 0.1]
        else:
            weights = [0.15, 0.35, 0.1, 0.2, 0.2]
        
        idx = self.rng.choices(range(len(collection_answers)), weights=weights, k=1)[0]
        answer, coll_type = collection_answers[idx]
        
        return answer, {
            "collection_type": coll_type,
            "total_count": count,
            "num_groups": num_groups,
            "group_sizes": group_sizes
        }

    def _answer_metaphor_arithmetic_construction(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Arithmetic is object construction (building/breaking = multiplication/division)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "nothing to construct or deconstruct", {"construction_type": "none"}
        
        masses = []
        for i in range(count):
            if active_mask[i]:
                mass = end_states[i, self.MASS_IDX].item() if end_states.shape[1] > self.MASS_IDX else 1.0
                masses.append(mass)
        
        total_mass = sum(masses)
        avg_mass = total_mass / len(masses)
        
        if len(masses) >= 2:
            ratio = max(masses) / min(masses) if min(masses) > 0.01 else 1
        else:
            ratio = 1
        
        construction_answers = [
            (f"Multiplication by {count}: {count} objects each of unit size = {count} total units", "multiplication"),
            (f"Division into {count} parts: a whole divided into {count} equal pieces", "division"),
            (f"Scaling by {ratio:.1f}: one object is {ratio:.1f} times the size of another", "scaling"),
            (f"Composite construction: {count} parts assembled into total mass {total_mass:.1f}", "composite"),
            (f"Factorization: total {total_mass:.1f} = product of {count} factors", "factorization"),
        ]
        
        if ratio > 2:
            weights = [0.15, 0.15, 0.4, 0.15, 0.15]
        elif count > 3:
            weights = [0.3, 0.25, 0.1, 0.2, 0.15]
        else:
            weights = [0.25, 0.25, 0.15, 0.2, 0.15]
        
        idx = self.rng.choices(range(len(construction_answers)), weights=weights, k=1)[0]
        answer, const_type = construction_answers[idx]
        
        return answer, {
            "construction_type": const_type,
            "count": count,
            "total_mass": total_mass,
            "mass_ratio": ratio
        }

    def _answer_metaphor_measuring_stick(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Numbers as measuring stick segments (physical lengths = numbers)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "single point - no distance to measure", {"measurement_type": "point"}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        
        distances = []
        for i in range(count):
            for j in range(i + 1, count):
                dist = torch.norm(positions[i] - positions[j]).item()
                distances.append((i + 1, j + 1, dist))
        
        distances.sort(key=lambda x: x[2])
        unit_length = distances[0][2] if distances[0][2] > 0.01 else 1.0
        
        measurements = [(d[0], d[1], d[2] / unit_length) for d in distances]
        
        measuring_answers = [
            (f"Unit distance between objects {measurements[0][0]} and {measurements[0][1]} defines 1 on our ruler", "unit_definition"),
            (f"Distance {measurements[-1][0]}-{measurements[-1][1]} measures {measurements[-1][2]:.1f} units", "max_measurement"),
            (f"Ruler readings: " + ", ".join([f"{m[0]}-{m[1]}={m[2]:.1f}" for m in measurements[:3]]), "multiple_measurements"),
            (f"Average spacing is {sum(d[2] for d in distances) / len(distances) / unit_length:.1f} units", "average_spacing"),
            (f"Scale: 1 unit = {unit_length:.2f} spatial units in this scene", "scale_definition"),
        ]
        
        idx = self.rng.randint(0, len(measuring_answers) - 1)
        answer, meas_type = measuring_answers[idx]
        
        return answer, {
            "measurement_type": meas_type,
            "unit_length": unit_length,
            "measurements": measurements,
            "num_distances": len(distances)
        }

    def _answer_metaphor_sets_containers(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Sets are containers (spatial regions = set membership)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "empty set - no elements", {"set_type": "empty"}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        center = positions.mean(dim=0)
        
        left_set = []
        right_set = []
        upper_set = []
        lower_set = []
        
        for i in range(count):
            pos = positions[i]
            obj_id = i + 1
            if pos[0].item() < center[0].item():
                left_set.append(obj_id)
            else:
                right_set.append(obj_id)
            if pos[1].item() > center[1].item():
                upper_set.append(obj_id)
            else:
                lower_set.append(obj_id)
        
        intersection_upper_left = set(left_set) & set(upper_set)
        union_all = set(range(1, count + 1))
        
        set_answers = [
            (f"Set A (left) = {{{', '.join(map(str, left_set))}}}, Set B (right) = {{{', '.join(map(str, right_set))}}}", "partition_lr"),
            (f"Set A (upper) = {{{', '.join(map(str, upper_set))}}}, Set B (lower) = {{{', '.join(map(str, lower_set))}}}", "partition_ud"),
            (f"Intersection of upper-left quadrant: {{{', '.join(map(str, intersection_upper_left))}}}", "intersection"),
            (f"Universal set U = {{{', '.join(map(str, union_all))}}} contains all {count} elements", "universal"),
            (f"Cardinality: |left|={len(left_set)}, |right|={len(right_set)}, |upper|={len(upper_set)}, |lower|={len(lower_set)}", "cardinality"),
        ]
        
        idx = self.rng.randint(0, len(set_answers) - 1)
        answer, set_type = set_answers[idx]
        
        return answer, {
            "set_type": set_type,
            "left_set": left_set,
            "right_set": right_set,
            "upper_set": upper_set,
            "lower_set": lower_set,
            "total_elements": count
        }

    def _answer_metaphor_continuity_gapless(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Continuity is gapless (smooth motion = continuous function)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no trajectory to analyze for continuity", {"continuity_type": "undefined"}
        
        vel = computed_vel[0]
        speed = torch.norm(vel).item()
        
        start_pos = start_states[0, self.POSITION_IDX]
        end_pos = end_states[0, self.POSITION_IDX]
        displacement = torch.norm(end_pos - start_pos).item()
        
        if speed < 0.001:
            smoothness = 1.0
            continuity_type = "constant"
        else:
            smoothness = min(1.0, displacement / (speed * 10 + 0.01))
            if smoothness > 0.8:
                continuity_type = "smooth"
            elif smoothness > 0.5:
                continuity_type = "piecewise"
            else:
                continuity_type = "discontinuous"
        
        continuity_answers = [
            ("Continuous function - smooth, gapless motion like f(x) with no jumps", "continuous"),
            ("Constant function - no change, f(x) = c, perfectly continuous but flat", "constant"),
            ("Piecewise continuous - mostly smooth with possible corner points", "piecewise"),
            ("Discontinuous - jumps or gaps in the trajectory, like a step function", "discontinuous"),
            (f"Smoothness index: {smoothness:.2f} - measuring how gapless the motion is", "smoothness_metric"),
        ]
        
        if continuity_type == "constant":
            weights = [0.1, 0.5, 0.1, 0.1, 0.2]
        elif continuity_type == "smooth":
            weights = [0.5, 0.1, 0.15, 0.05, 0.2]
        elif continuity_type == "piecewise":
            weights = [0.15, 0.1, 0.4, 0.15, 0.2]
        else:
            weights = [0.1, 0.05, 0.15, 0.5, 0.2]
        
        idx = self.rng.choices(range(len(continuity_answers)), weights=weights, k=1)[0]
        answer, cont_type = continuity_answers[idx]
        
        return answer, {
            "continuity_type": cont_type,
            "smoothness": smoothness,
            "speed": speed,
            "displacement": displacement
        }

    def _answer_metaphor_change_motion(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Change is motion (rate of change = velocity, derivative = speed)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no change to measure - static system", {"change_type": "static"}
        
        speeds = []
        accelerations = []
        for i in range(count):
            if active_mask[i]:
                vel = computed_vel[i]
                speed = torch.norm(vel).item()
                speeds.append(speed)
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        total_change = sum(speeds)
        
        change_answers = [
            (f"Rate of change (derivative) = {avg_speed:.3f} - the system changes at this velocity", "derivative"),
            (f"Maximum rate: {max_speed:.3f} - the fastest changing element sets the pace", "max_rate"),
            (f"Total flux: {total_change:.3f} - sum of all rates of change in the system", "total_flux"),
            ("Near-zero derivative - the system is at a critical point or equilibrium", "critical_point"),
            (f"Gradient: change flows at {avg_speed:.3f} units per time step", "gradient"),
            ("Rapid change - high derivatives indicate fast-moving dynamics", "rapid"),
            ("Slow change - small derivatives mean gradual evolution", "slow"),
        ]
        
        if avg_speed < 0.001:
            weights = [0.1, 0.05, 0.1, 0.5, 0.1, 0.05, 0.1]
        elif avg_speed > 0.1:
            weights = [0.2, 0.2, 0.15, 0.02, 0.15, 0.25, 0.03]
        else:
            weights = [0.25, 0.15, 0.15, 0.05, 0.15, 0.1, 0.15]
        
        idx = self.rng.choices(range(len(change_answers)), weights=weights, k=1)[0]
        answer, change_type = change_answers[idx]
        
        return answer, {
            "change_type": change_type,
            "avg_rate": avg_speed,
            "max_rate": max_speed,
            "total_flux": total_change
        }

    def _answer_metaphor_numbers_points(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Numbers are points on a line (positions = numerical values)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no points on the number line", {"number_type": "empty"}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        
        x_coords = [positions[i, 0].item() for i in range(count)]
        y_coords = [positions[i, 1].item() for i in range(count)] if positions.shape[1] > 1 else [0] * count
        
        x_coords_sorted = sorted(enumerate(x_coords), key=lambda x: x[1])
        
        number_line = [(i + 1, x) for i, x in x_coords_sorted]
        
        min_val = min(x_coords)
        max_val = max(x_coords)
        range_val = max_val - min_val
        
        numbers_answers = [
            (f"Number line: " + " < ".join([f"obj{n[0]}({n[1]:.2f})" for n in number_line]), "ordered_line"),
            (f"Range [{min_val:.2f}, {max_val:.2f}] - numbers span {range_val:.2f} units", "range"),
            (f"Object 1 at coordinate ({x_coords[0]:.2f}, {y_coords[0]:.2f}) - a point in 2D number space", "coordinate"),
            (f"Minimum value {min_val:.2f} at object {x_coords.index(min_val) + 1}", "minimum"),
            (f"Maximum value {max_val:.2f} at object {x_coords.index(max_val) + 1}", "maximum"),
            (f"Mean position: {sum(x_coords) / len(x_coords):.2f} - the average number", "mean"),
        ]
        
        idx = self.rng.randint(0, len(numbers_answers) - 1)
        answer, num_type = numbers_answers[idx]
        
        return answer, {
            "number_type": num_type,
            "x_coordinates": x_coords,
            "y_coordinates": y_coords,
            "range": range_val,
            "min": min_val,
            "max": max_val
        }

    def _answer_metaphor_recurrence_circular(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Recurrence is circular (periodic patterns = circular motion)."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no pattern to analyze for recurrence", {"recurrence_type": "none"}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        velocities = computed_vel[active_mask] if computed_vel.dim() > 1 else computed_vel.unsqueeze(0)
        
        center = positions.mean(dim=0)
        
        radii = torch.norm(positions - center, dim=1)
        avg_radius = radii.mean().item()
        radius_variance = radii.std().item() if count > 1 else 0
        
        tangential_components = []
        for i in range(min(count, velocities.shape[0])):
            pos = positions[i] - center
            vel = velocities[i]
            radial_dir = pos / (torch.norm(pos) + 1e-6)
            radial_vel = (vel * radial_dir).sum()
            tangential_vel = torch.norm(vel - radial_vel * radial_dir).item()
            tangential_components.append(tangential_vel)
        
        avg_tangential = sum(tangential_components) / len(tangential_components) if tangential_components else 0
        
        is_circular = radius_variance < avg_radius * 0.3 and avg_tangential > 0.01
        
        recurrence_answers = [
            (f"Circular orbit detected - objects revolve at radius {avg_radius:.2f}, period proportional to circumference", "circular"),
            ("Elliptical cycle - elongated periodic motion, like planetary orbits", "elliptical"),
            ("Spiral pattern - circular with changing radius, converging or diverging", "spiral"),
            ("No clear cycle - motion is not periodic, no recurrence detected", "acyclic"),
            (f"Oscillation - back-and-forth motion with period related to {2 * 3.14159 * avg_radius / (avg_tangential + 0.01):.1f} time units", "oscillation"),
            ("Quasi-periodic - almost cyclic but with drift, like a precessing orbit", "quasi_periodic"),
        ]
        
        if is_circular:
            weights = [0.4, 0.25, 0.1, 0.05, 0.1, 0.1]
        elif avg_tangential > 0.01:
            weights = [0.15, 0.2, 0.2, 0.1, 0.2, 0.15]
        else:
            weights = [0.05, 0.1, 0.1, 0.5, 0.15, 0.1]
        
        idx = self.rng.choices(range(len(recurrence_answers)), weights=weights, k=1)[0]
        answer, rec_type = recurrence_answers[idx]
        
        return answer, {
            "recurrence_type": rec_type,
            "avg_radius": avg_radius,
            "radius_variance": radius_variance,
            "avg_tangential_velocity": avg_tangential,
            "is_circular": is_circular
        }

    def _answer_metaphor_infinity(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Answer: Basic Metaphor of Infinity (BMI) - limits and infinite continuation."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "no process to extend to infinity", {"infinity_type": "undefined"}
        
        vel = computed_vel[0]
        speed = torch.norm(vel).item()
        pos = end_states[0, self.POSITION_IDX]
        
        if speed > 0.001:
            direction = vel / torch.norm(vel)
            limit_direction = direction.tolist()
            projected_limit = pos + vel * 1000
            limit_magnitude = torch.norm(projected_limit).item()
        else:
            limit_direction = [0, 0, 0]
            limit_magnitude = torch.norm(pos).item()
        
        is_bounded = speed < 0.01
        is_converging = speed > 0.001 and torch.norm(pos).item() > torch.norm(pos + vel * 10).item()
        
        infinity_answers = [
            (f"Limit as t→∞: position approaches infinity in direction {[f'{d:.2f}' for d in limit_direction[:2]]}", "divergent"),
            ("Bounded limit - the process stays finite, converging to a fixed point", "convergent"),
            ("Potential infinite - the process can continue indefinitely but never reaches infinity", "potential"),
            ("Actual infinite - if we imagine completing infinitely many steps, we reach a limit", "actual"),
            (f"Asymptotic behavior - approaching but never reaching the limit at magnitude {limit_magnitude:.1f}", "asymptotic"),
            ("Oscillating limit - the process doesn't settle, alternating forever", "oscillating"),
        ]
        
        if is_bounded:
            weights = [0.05, 0.4, 0.2, 0.15, 0.15, 0.05]
        elif is_converging:
            weights = [0.1, 0.3, 0.15, 0.2, 0.2, 0.05]
        elif speed > 0.1:
            weights = [0.4, 0.05, 0.2, 0.15, 0.15, 0.05]
        else:
            weights = [0.2, 0.15, 0.25, 0.15, 0.15, 0.1]
        
        idx = self.rng.choices(range(len(infinity_answers)), weights=weights, k=1)[0]
        answer, inf_type = infinity_answers[idx]
        
        return answer, {
            "infinity_type": inf_type,
            "speed": speed,
            "is_bounded": is_bounded,
            "is_converging": is_converging,
            "limit_direction": limit_direction
        }

    # =========================================================================
    # CATEGORY 5: Evaluative Concepts (Safety & Normative Judgments)
    # These ground abstract evaluative concepts in physical states
    # =========================================================================

    def _answer_safety_assessment(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess overall safety based on physical configuration."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "Safe - no active objects present", {"safety_level": "safe", "risk_score": 0.0}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        velocities = computed_vel[active_mask]
        speeds = torch.norm(velocities, dim=-1)
        
        min_dist = float('inf')
        for i in range(count):
            for j in range(i + 1, count):
                dist = torch.norm(positions[i] - positions[j]).item()
                min_dist = min(min_dist, dist)
        
        if min_dist == float('inf'):
            min_dist = 10.0
        
        avg_speed = speeds.mean().item()
        max_speed = speeds.max().item() if count > 0 else 0
        
        risk_score = (max_speed * 0.3 + (1.0 / (min_dist + 0.1)) * 0.7)
        risk_score = min(1.0, risk_score)
        
        safety_answers = [
            ("Safe - objects are well-separated and moving slowly", "safe"),
            ("Generally safe - low risk of collision, normal operating conditions", "low_risk"),
            ("Caution advised - moderate speeds and proximity require attention", "moderate"),
            ("Elevated risk - fast-moving objects in close proximity", "elevated"),
            ("Dangerous - high collision probability, immediate attention required", "dangerous"),
            ("Critical - imminent collision, emergency response needed", "critical"),
        ]
        
        if risk_score < 0.1:
            weights = [0.5, 0.35, 0.1, 0.03, 0.01, 0.01]
        elif risk_score < 0.3:
            weights = [0.2, 0.4, 0.25, 0.1, 0.03, 0.02]
        elif risk_score < 0.5:
            weights = [0.05, 0.15, 0.4, 0.25, 0.1, 0.05]
        elif risk_score < 0.7:
            weights = [0.02, 0.05, 0.15, 0.4, 0.28, 0.1]
        else:
            weights = [0.01, 0.02, 0.05, 0.15, 0.37, 0.4]
        
        idx = self.rng.choices(range(len(safety_answers)), weights=weights, k=1)[0]
        answer, safety_level = safety_answers[idx]
        
        return answer, {
            "safety_level": safety_level,
            "risk_score": risk_score,
            "min_distance": min_dist,
            "max_speed": max_speed,
            "avg_speed": avg_speed
        }

    def _answer_danger_level(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Rate the danger level of the configuration."""
        answer, metadata = self._answer_safety_assessment(states, mask)
        
        danger_map = {
            "safe": "Minimal danger - situation is under control",
            "low_risk": "Low danger - minor risks present but manageable",
            "moderate": "Moderate danger - requires monitoring and caution",
            "elevated": "High danger - significant risk of harm",
            "dangerous": "Severe danger - immediate risk to safety",
            "critical": "Extreme danger - catastrophic outcome likely without intervention"
        }
        
        danger_answer = danger_map.get(metadata["safety_level"], "Unknown danger level")
        metadata["danger_level"] = metadata["safety_level"]
        
        return danger_answer, metadata

    def _answer_threat_to_self(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess if the primary agent (object 1) is in danger."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No threat - agent not present", {"threat_level": "none", "threat_score": 0.0}
        
        agent_pos = end_states[0, self.POSITION_IDX]
        agent_vel = computed_vel[0]
        
        threat_score = 0.0
        threatening_objects = []
        
        for i in range(1, count):
            if not active_mask[i]:
                continue
            
            other_pos = end_states[i, self.POSITION_IDX]
            other_vel = computed_vel[i]
            
            relative_pos = other_pos - agent_pos
            relative_vel = other_vel - agent_vel
            
            distance = torch.norm(relative_pos).item()
            closing_speed = -torch.dot(relative_pos, relative_vel).item() / (distance + 0.01)
            
            if closing_speed > 0:
                time_to_contact = distance / (closing_speed + 0.01)
                if time_to_contact < 10:
                    obj_threat = closing_speed / (distance + 0.1)
                    threat_score += obj_threat
                    threatening_objects.append(i)
        
        threat_score = min(1.0, threat_score)
        
        threat_answers = [
            ("No immediate threat - agent is safe", "none"),
            ("Low threat - distant objects, no immediate concern", "low"),
            ("Moderate threat - approaching objects detected, stay alert", "moderate"),
            ("High threat - fast-approaching object, evasive action recommended", "high"),
            ("Imminent danger - collision likely, take immediate action", "imminent"),
        ]
        
        if threat_score < 0.1:
            weights = [0.6, 0.3, 0.07, 0.02, 0.01]
        elif threat_score < 0.3:
            weights = [0.2, 0.45, 0.25, 0.08, 0.02]
        elif threat_score < 0.5:
            weights = [0.05, 0.15, 0.45, 0.25, 0.1]
        elif threat_score < 0.7:
            weights = [0.02, 0.05, 0.15, 0.45, 0.33]
        else:
            weights = [0.01, 0.02, 0.07, 0.3, 0.6]
        
        idx = self.rng.choices(range(len(threat_answers)), weights=weights, k=1)[0]
        answer, threat_level = threat_answers[idx]
        
        return answer, {
            "threat_level": threat_level,
            "threat_score": threat_score,
            "threatening_objects": threatening_objects,
            "num_threats": len(threatening_objects)
        }

    def _answer_threat_to_others(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess if the primary agent is endangering other objects."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "No others to endanger", {"endangering": False, "endangered_count": 0}
        
        agent_pos = end_states[0, self.POSITION_IDX]
        agent_vel = computed_vel[0]
        agent_speed = torch.norm(agent_vel).item()
        agent_mass = end_states[0, self.MASS_IDX].item() if end_states.shape[-1] > self.MASS_IDX else 1.0
        
        endangered_objects = []
        
        for i in range(1, count):
            if not active_mask[i]:
                continue
            
            other_pos = end_states[i, self.POSITION_IDX]
            relative_pos = other_pos - agent_pos
            distance = torch.norm(relative_pos).item()
            
            if agent_speed > 0.01:
                direction = agent_vel / torch.norm(agent_vel)
                alignment = torch.dot(relative_pos / (distance + 0.01), direction).item()
                
                if alignment > 0.5 and distance < agent_speed * 10:
                    endangered_objects.append(i)
        
        endangering = len(endangered_objects) > 0
        
        if not endangering:
            answer = "Not endangering others - agent's trajectory is clear of other objects"
        elif len(endangered_objects) == 1:
            answer = f"Potentially endangering object {endangered_objects[0] + 1} - agent is moving toward it"
        else:
            answer = f"Endangering {len(endangered_objects)} objects - agent's trajectory intersects multiple objects"
        
        return answer, {
            "endangering": endangering,
            "endangered_objects": endangered_objects,
            "endangered_count": len(endangered_objects),
            "agent_speed": agent_speed,
            "agent_mass": agent_mass
        }

    def _answer_stability_assessment(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess the structural stability of the configuration."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "Stable - no objects to assess", {"stability": "stable", "stability_score": 1.0}
        
        velocities = computed_vel[active_mask]
        speeds = torch.norm(velocities, dim=-1)
        avg_speed = speeds.mean().item()
        
        positions = end_states[active_mask, self.POSITION_IDX]
        center_of_mass = positions.mean(dim=0)
        spread = torch.norm(positions - center_of_mass, dim=-1).mean().item()
        
        grounded_count = sum(1 for i in range(count) if active_mask[i] and 
                            end_states[i, self.POSITION_IDX][2].item() < 0.5)
        
        stability_score = 1.0 - min(1.0, avg_speed * 2 + (1.0 - grounded_count / max(count, 1)) * 0.5)
        
        stability_answers = [
            ("Highly stable - objects at rest, well-grounded configuration", "highly_stable"),
            ("Stable - minimal motion, balanced arrangement", "stable"),
            ("Marginally stable - some motion but maintaining structure", "marginal"),
            ("Unstable - significant motion, configuration changing", "unstable"),
            ("Highly unstable - rapid motion, structure collapsing", "highly_unstable"),
        ]
        
        if stability_score > 0.8:
            weights = [0.5, 0.35, 0.1, 0.03, 0.02]
        elif stability_score > 0.6:
            weights = [0.15, 0.45, 0.3, 0.08, 0.02]
        elif stability_score > 0.4:
            weights = [0.05, 0.15, 0.45, 0.25, 0.1]
        elif stability_score > 0.2:
            weights = [0.02, 0.05, 0.15, 0.48, 0.3]
        else:
            weights = [0.01, 0.02, 0.07, 0.3, 0.6]
        
        idx = self.rng.choices(range(len(stability_answers)), weights=weights, k=1)[0]
        answer, stability = stability_answers[idx]
        
        return answer, {
            "stability": stability,
            "stability_score": stability_score,
            "avg_speed": avg_speed,
            "grounded_ratio": grounded_count / max(count, 1),
            "spread": spread
        }

    def _answer_collision_risk(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess the probability of collision."""
        answer, metadata = self._answer_safety_assessment(states, mask)
        
        risk_score = metadata["risk_score"]
        
        risk_answers = [
            ("Negligible collision risk - objects well-separated", "negligible"),
            ("Low collision risk - safe distances maintained", "low"),
            ("Moderate collision risk - trajectories may intersect", "moderate"),
            ("High collision risk - objects on converging paths", "high"),
            ("Collision imminent - impact expected within seconds", "imminent"),
        ]
        
        if risk_score < 0.15:
            weights = [0.6, 0.3, 0.07, 0.02, 0.01]
        elif risk_score < 0.35:
            weights = [0.15, 0.5, 0.25, 0.08, 0.02]
        elif risk_score < 0.55:
            weights = [0.05, 0.15, 0.5, 0.22, 0.08]
        elif risk_score < 0.75:
            weights = [0.02, 0.05, 0.15, 0.5, 0.28]
        else:
            weights = [0.01, 0.02, 0.07, 0.25, 0.65]
        
        idx = self.rng.choices(range(len(risk_answers)), weights=weights, k=1)[0]
        risk_answer, risk_level = risk_answers[idx]
        
        metadata["collision_risk"] = risk_level
        return risk_answer, metadata

    def _answer_escape_routes(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Identify available escape routes for the agent."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "All directions open - no obstacles", {"escape_routes": ["all"], "blocked_directions": []}
        
        agent_pos = end_states[0, self.POSITION_IDX]
        
        directions = {
            "left": torch.tensor([-1.0, 0.0, 0.0]),
            "right": torch.tensor([1.0, 0.0, 0.0]),
            "forward": torch.tensor([0.0, 1.0, 0.0]),
            "backward": torch.tensor([0.0, -1.0, 0.0]),
            "up": torch.tensor([0.0, 0.0, 1.0]),
        }
        
        blocked = []
        clear = []
        
        for dir_name, dir_vec in directions.items():
            is_blocked = False
            for i in range(1, count):
                if not active_mask[i]:
                    continue
                other_pos = end_states[i, self.POSITION_IDX]
                relative = other_pos - agent_pos
                distance = torch.norm(relative).item()
                if distance < 3.0:
                    alignment = torch.dot(relative / (distance + 0.01), dir_vec).item()
                    if alignment > 0.7:
                        is_blocked = True
                        break
            
            if is_blocked:
                blocked.append(dir_name)
            else:
                clear.append(dir_name)
        
        if len(clear) == 0:
            answer = "No clear escape routes - surrounded by obstacles"
        elif len(clear) == len(directions):
            answer = "All escape routes clear - full freedom of movement"
        else:
            answer = f"Escape routes available: {', '.join(clear)}. Blocked: {', '.join(blocked)}"
        
        return answer, {
            "escape_routes": clear,
            "blocked_directions": blocked,
            "num_clear": len(clear),
            "num_blocked": len(blocked)
        }

    def _answer_protective_action(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Recommend action to increase safety."""
        escape_answer, escape_meta = self._answer_escape_routes(states, mask)
        threat_answer, threat_meta = self._answer_threat_to_self(states, mask)
        
        clear_routes = escape_meta["escape_routes"]
        threat_level = threat_meta["threat_level"]
        
        if threat_level in ["none", "low"]:
            action = "Maintain current course - no protective action needed"
            recommended = "none"
        elif len(clear_routes) > 0:
            best_route = clear_routes[0]
            action = f"Move {best_route} to increase safety margin"
            recommended = f"move_{best_route}"
        else:
            action = "Reduce speed and prepare for impact - no clear escape route"
            recommended = "brace"
        
        return action, {
            "recommended_action": recommended,
            "clear_routes": clear_routes,
            "threat_level": threat_level
        }

    def _answer_vulnerability(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Identify the most vulnerable objects."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No objects to assess for vulnerability", {"vulnerable_objects": []}
        
        vulnerabilities = []
        
        for i in range(count):
            if not active_mask[i]:
                continue
            
            pos = end_states[i, self.POSITION_IDX]
            vel = computed_vel[i]
            speed = torch.norm(vel).item()
            mass = end_states[i, self.MASS_IDX].item() if end_states.shape[-1] > self.MASS_IDX else 1.0
            
            incoming_threats = 0
            for j in range(count):
                if i == j or not active_mask[j]:
                    continue
                other_pos = end_states[j, self.POSITION_IDX]
                other_vel = computed_vel[j]
                relative_pos = pos - other_pos
                distance = torch.norm(relative_pos).item()
                
                if distance < 5.0:
                    closing = torch.dot(other_vel, relative_pos / (distance + 0.01)).item()
                    if closing > 0:
                        incoming_threats += 1
            
            vulnerability_score = incoming_threats / max(mass, 0.1) + (1.0 - speed) * 0.1
            vulnerabilities.append((i, vulnerability_score))
        
        vulnerabilities.sort(key=lambda x: x[1], reverse=True)
        most_vulnerable = [v[0] for v in vulnerabilities[:3]]
        
        if len(most_vulnerable) == 0:
            answer = "No particularly vulnerable objects identified"
        else:
            answer = f"Most vulnerable: object {most_vulnerable[0] + 1}" + \
                    (f" (also objects {', '.join(str(o+1) for o in most_vulnerable[1:])})" if len(most_vulnerable) > 1 else "")
        
        return answer, {
            "vulnerable_objects": most_vulnerable,
            "vulnerability_scores": dict(vulnerabilities)
        }

    def _answer_causal_responsibility(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Identify what would cause harm if it occurs."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "No potential for harm - insufficient objects", {"responsible_object": None}
        
        max_danger = 0
        responsible = None
        
        for i in range(count):
            if not active_mask[i]:
                continue
            
            vel = computed_vel[i]
            speed = torch.norm(vel).item()
            mass = end_states[i, self.MASS_IDX].item() if end_states.shape[-1] > self.MASS_IDX else 1.0
            momentum = speed * mass
            
            pos = end_states[i, self.POSITION_IDX]
            targets_in_path = 0
            
            for j in range(count):
                if i == j or not active_mask[j]:
                    continue
                other_pos = end_states[j, self.POSITION_IDX]
                relative = other_pos - pos
                distance = torch.norm(relative).item()
                
                if speed > 0.01 and distance < 5.0:
                    alignment = torch.dot(vel / speed, relative / (distance + 0.01)).item()
                    if alignment > 0.5:
                        targets_in_path += 1
            
            danger_potential = momentum * (targets_in_path + 1)
            if danger_potential > max_danger:
                max_danger = danger_potential
                responsible = i
        
        if responsible is None:
            answer = "No clear source of danger identified"
        else:
            answer = f"Object {responsible + 1} would be primarily responsible - highest momentum toward other objects"
        
        return answer, {
            "responsible_object": responsible,
            "danger_potential": max_danger
        }

    def _answer_force_assessment(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess force level: gentle, moderate, firm, or crushing."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No force to assess - no active objects", {"force_level": "none", "force_magnitude": 0.0}
        
        velocities = computed_vel[active_mask]
        speeds = torch.norm(velocities, dim=-1)
        
        masses = []
        for i in range(count):
            if active_mask[i] and end_states.shape[-1] > self.MASS_IDX:
                masses.append(end_states[i, self.MASS_IDX].item())
            else:
                masses.append(1.0)
        
        momenta = [speeds[i].item() * masses[i] for i in range(len(masses))]
        max_momentum = max(momenta) if momenta else 0
        
        min_dist = float('inf')
        closing_momentum = 0
        for i in range(count):
            for j in range(i + 1, count):
                if active_mask[i] and active_mask[j]:
                    pos_i = end_states[i, self.POSITION_IDX]
                    pos_j = end_states[j, self.POSITION_IDX]
                    dist = torch.norm(pos_i - pos_j).item()
                    if dist < min_dist:
                        min_dist = dist
                        rel_vel = velocities[i] - velocities[j]
                        closing_momentum = torch.norm(rel_vel).item() * (masses[i] + masses[j]) / 2
        
        if min_dist == float('inf'):
            min_dist = 10.0
        
        force_estimate = closing_momentum / (min_dist + 0.1)
        
        force_answers = [
            ("Gentle - minimal force, safe for delicate objects", "gentle"),
            ("Light - low force, suitable for careful handling", "light"),
            ("Moderate - noticeable force, standard interaction", "moderate"),
            ("Firm - significant force, requires sturdy objects", "firm"),
            ("Forceful - high force, risk of damage to fragile items", "forceful"),
            ("Crushing - extreme force, likely to cause damage", "crushing"),
        ]
        
        if force_estimate < 0.05:
            weights = [0.5, 0.3, 0.12, 0.05, 0.02, 0.01]
        elif force_estimate < 0.15:
            weights = [0.2, 0.4, 0.25, 0.1, 0.03, 0.02]
        elif force_estimate < 0.3:
            weights = [0.05, 0.15, 0.4, 0.25, 0.1, 0.05]
        elif force_estimate < 0.5:
            weights = [0.02, 0.05, 0.15, 0.4, 0.28, 0.1]
        elif force_estimate < 0.8:
            weights = [0.01, 0.02, 0.08, 0.2, 0.44, 0.25]
        else:
            weights = [0.01, 0.01, 0.03, 0.1, 0.3, 0.55]
        
        idx = self.rng.choices(range(len(force_answers)), weights=weights, k=1)[0]
        answer, force_level = force_answers[idx]
        
        return answer, {
            "force_level": force_level,
            "force_magnitude": force_estimate,
            "max_momentum": max_momentum,
            "min_distance": min_dist,
            "closing_momentum": closing_momentum
        }

    def _answer_structural_load(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess if configuration can support the weight/load."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No load to assess", {"load_status": "empty", "load_ratio": 0.0}
        
        positions = end_states[active_mask, self.POSITION_IDX]
        heights = positions[:, 2] if positions.shape[-1] > 2 else positions[:, 1]
        
        grounded_objects = []
        elevated_objects = []
        
        for i in range(count):
            if not active_mask[i]:
                continue
            height = heights[i].item() if i < len(heights) else 0
            mass = end_states[i, self.MASS_IDX].item() if end_states.shape[-1] > self.MASS_IDX else 1.0
            
            if height < 0.5:
                grounded_objects.append((i, mass))
            else:
                elevated_objects.append((i, mass, height))
        
        total_grounded_mass = sum(m for _, m in grounded_objects)
        total_elevated_mass = sum(m for _, m, _ in elevated_objects)
        
        if total_grounded_mass > 0:
            load_ratio = total_elevated_mass / total_grounded_mass
        else:
            load_ratio = total_elevated_mass if total_elevated_mass > 0 else 0
        
        velocities = computed_vel[active_mask]
        avg_speed = torch.norm(velocities, dim=-1).mean().item()
        
        stability_factor = 1.0 - min(1.0, avg_speed * 5)
        adjusted_load = load_ratio / (stability_factor + 0.1)
        
        load_answers = [
            ("Fully supported - load well within capacity, stable configuration", "fully_supported"),
            ("Adequately supported - load is manageable, minor stress", "adequate"),
            ("Marginally supported - near capacity, caution advised", "marginal"),
            ("Overloaded - exceeding safe capacity, risk of failure", "overloaded"),
            ("Critical overload - structural failure imminent", "critical"),
            ("Unsupported - elevated mass with no base support", "unsupported"),
        ]
        
        if len(grounded_objects) == 0 and len(elevated_objects) > 0:
            weights = [0.01, 0.02, 0.05, 0.12, 0.2, 0.6]
        elif adjusted_load < 0.3:
            weights = [0.5, 0.35, 0.1, 0.03, 0.01, 0.01]
        elif adjusted_load < 0.6:
            weights = [0.2, 0.4, 0.28, 0.08, 0.03, 0.01]
        elif adjusted_load < 1.0:
            weights = [0.05, 0.15, 0.45, 0.25, 0.08, 0.02]
        elif adjusted_load < 1.5:
            weights = [0.02, 0.05, 0.15, 0.45, 0.28, 0.05]
        else:
            weights = [0.01, 0.02, 0.07, 0.2, 0.5, 0.2]
        
        idx = self.rng.choices(range(len(load_answers)), weights=weights, k=1)[0]
        answer, load_status = load_answers[idx]
        
        return answer, {
            "load_status": load_status,
            "load_ratio": load_ratio,
            "grounded_mass": total_grounded_mass,
            "elevated_mass": total_elevated_mass,
            "stability_factor": stability_factor,
            "num_grounded": len(grounded_objects),
            "num_elevated": len(elevated_objects)
        }

    def _answer_urgency_assessment(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess how urgent action is required."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No urgency - no active situation", {"urgency_level": "none", "time_pressure": 0.0}
        
        threat_answer, threat_meta = self._answer_threat_to_self(states, mask)
        collision_answer, collision_meta = self._answer_collision_risk(states, mask)
        stability_answer, stability_meta = self._answer_stability_assessment(states, mask)
        
        threat_score = threat_meta.get("threat_score", 0)
        risk_score = collision_meta.get("risk_score", 0)
        stability_score = stability_meta.get("stability_score", 1.0)
        
        instability = 1.0 - stability_score
        
        time_pressure = (threat_score * 0.4 + risk_score * 0.4 + instability * 0.2)
        time_pressure = min(1.0, time_pressure)
        
        min_time_to_event = float('inf')
        positions = end_states[active_mask, self.POSITION_IDX]
        velocities = computed_vel[active_mask]
        
        for i in range(count):
            for j in range(i + 1, count):
                if active_mask[i] and active_mask[j]:
                    rel_pos = positions[j] - positions[i]
                    rel_vel = velocities[j] - velocities[i]
                    distance = torch.norm(rel_pos).item()
                    closing_speed = -torch.dot(rel_pos, rel_vel).item() / (distance + 0.01)
                    
                    if closing_speed > 0.01:
                        time_to_contact = distance / closing_speed
                        min_time_to_event = min(min_time_to_event, time_to_contact)
        
        if min_time_to_event < float('inf'):
            time_pressure = max(time_pressure, 1.0 / (min_time_to_event + 1))
        
        urgency_answers = [
            ("No urgency - situation is stable, take your time", "none"),
            ("Low urgency - can respond at normal pace", "low"),
            ("Moderate urgency - should act soon but not immediately", "moderate"),
            ("High urgency - prompt action recommended", "high"),
            ("Urgent - immediate action required", "urgent"),
            ("Critical - act now, no time to delay", "critical"),
        ]
        
        if time_pressure < 0.1:
            weights = [0.5, 0.35, 0.1, 0.03, 0.01, 0.01]
        elif time_pressure < 0.25:
            weights = [0.15, 0.45, 0.28, 0.08, 0.03, 0.01]
        elif time_pressure < 0.45:
            weights = [0.05, 0.15, 0.45, 0.25, 0.08, 0.02]
        elif time_pressure < 0.65:
            weights = [0.02, 0.05, 0.15, 0.45, 0.25, 0.08]
        elif time_pressure < 0.85:
            weights = [0.01, 0.02, 0.08, 0.2, 0.45, 0.24]
        else:
            weights = [0.01, 0.01, 0.03, 0.1, 0.3, 0.55]
        
        idx = self.rng.choices(range(len(urgency_answers)), weights=weights, k=1)[0]
        answer, urgency_level = urgency_answers[idx]
        
        return answer, {
            "urgency_level": urgency_level,
            "time_pressure": time_pressure,
            "threat_score": threat_score,
            "risk_score": risk_score,
            "stability_score": stability_score,
            "min_time_to_event": min_time_to_event if min_time_to_event < float('inf') else None
        }

    # =========================================================================
    # CATEGORY 6: Intentional Concepts (Agency & Goal Attribution)
    # =========================================================================

    def _answer_agent_identification(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Identify objects that appear to be agents with goals."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No objects to assess for agency", {"agents": []}
        
        agents = []
        
        for i in range(count):
            if not active_mask[i]:
                continue
            
            vel = computed_vel[i]
            speed = torch.norm(vel).item()
            
            start_vel = start_states[i, self.VELOCITY_IDX] if start_states.shape[-1] > 5 else torch.zeros(3)
            vel_change = torch.norm(vel - start_vel).item()
            
            if speed > 0.05 or vel_change > 0.02:
                agents.append(i)
        
        if len(agents) == 0:
            answer = "No apparent agents - all objects appear passive or stationary"
        elif len(agents) == 1:
            answer = f"Object {agents[0] + 1} appears to be an agent - showing purposeful motion"
        else:
            answer = f"Multiple agents detected: objects {', '.join(str(a+1) for a in agents)}"
        
        return answer, {
            "agents": agents,
            "num_agents": len(agents)
        }

    def _answer_goal_inference(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Infer the goal of the primary moving object."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No object to infer goals from", {"inferred_goal": "none"}
        
        agent_pos = end_states[0, self.POSITION_IDX]
        agent_vel = computed_vel[0]
        speed = torch.norm(agent_vel).item()
        
        if speed < 0.01:
            return "Object appears stationary - goal may be to remain in place", {"inferred_goal": "stay"}
        
        direction = agent_vel / speed
        
        closest_target = None
        min_alignment_dist = float('inf')
        
        for i in range(1, count):
            if not active_mask[i]:
                continue
            other_pos = end_states[i, self.POSITION_IDX]
            relative = other_pos - agent_pos
            distance = torch.norm(relative).item()
            
            if distance > 0.1:
                alignment = torch.dot(direction, relative / distance).item()
                if alignment > 0.7:
                    alignment_dist = distance * (1 - alignment)
                    if alignment_dist < min_alignment_dist:
                        min_alignment_dist = alignment_dist
                        closest_target = i
        
        goal_answers = [
            (f"Approaching object {closest_target + 1} - appears to be the target", "approach") if closest_target else ("Moving toward open space", "explore"),
            ("Seeking a specific location - directed motion pattern", "seek_location"),
            ("Patrolling or exploring - no specific target apparent", "patrol"),
            ("Intercepting another object's path", "intercept"),
            ("Escaping or avoiding - moving away from threats", "escape"),
        ]
        
        if closest_target is not None:
            weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        elif speed > 0.1:
            weights = [0.1, 0.25, 0.3, 0.15, 0.2]
        else:
            weights = [0.15, 0.2, 0.35, 0.15, 0.15]
        
        idx = self.rng.choices(range(len(goal_answers)), weights=weights, k=1)[0]
        answer, goal = goal_answers[idx]
        
        return answer, {
            "inferred_goal": goal,
            "target_object": closest_target,
            "speed": speed
        }

    def _answer_helping_hindering(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess whether object 2 is helping or hindering object 1."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "Insufficient objects to assess interaction", {"interaction": "none"}
        
        agent_pos = end_states[0, self.POSITION_IDX]
        agent_vel = computed_vel[0]
        agent_speed = torch.norm(agent_vel).item()
        
        other_pos = end_states[1, self.POSITION_IDX]
        other_vel = computed_vel[1]
        
        relative_pos = other_pos - agent_pos
        distance = torch.norm(relative_pos).item()
        
        if agent_speed < 0.01:
            return "Agent stationary - cannot determine helping/hindering", {"interaction": "neutral"}
        
        agent_dir = agent_vel / agent_speed
        
        in_path = torch.dot(agent_dir, relative_pos / (distance + 0.01)).item() > 0.5 and distance < 5.0
        
        same_direction = torch.dot(agent_vel, other_vel).item() > 0
        
        if in_path and not same_direction:
            answer = "Object 2 is hindering - blocking the agent's path"
            interaction = "hindering"
        elif same_direction and distance < 3.0:
            answer = "Object 2 appears to be helping - moving in same direction, possibly guiding"
            interaction = "helping"
        elif in_path:
            answer = "Object 2 is a potential obstacle - in the path but not actively blocking"
            interaction = "obstacle"
        else:
            answer = "Object 2 is neutral - not significantly affecting the agent"
            interaction = "neutral"
        
        return answer, {
            "interaction": interaction,
            "in_path": in_path,
            "same_direction": same_direction,
            "distance": distance
        }

    def _answer_chasing_fleeing(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Characterize pursuit dynamics between objects."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "Insufficient objects for chase/flee analysis", {"dynamic": "none"}
        
        pos1 = end_states[0, self.POSITION_IDX]
        vel1 = computed_vel[0]
        speed1 = torch.norm(vel1).item()
        
        pos2 = end_states[1, self.POSITION_IDX]
        vel2 = computed_vel[1]
        speed2 = torch.norm(vel2).item()
        
        relative_pos = pos2 - pos1
        distance = torch.norm(relative_pos).item()
        
        if speed1 < 0.01 and speed2 < 0.01:
            return "Both objects stationary - no chase or flee dynamics", {"dynamic": "stationary"}
        
        obj1_toward_obj2 = torch.dot(vel1, relative_pos / (distance + 0.01)).item() if speed1 > 0.01 else 0
        obj2_away_from_obj1 = torch.dot(vel2, relative_pos / (distance + 0.01)).item() if speed2 > 0.01 else 0
        
        if obj1_toward_obj2 > 0.5 and obj2_away_from_obj1 > 0.3:
            answer = "Object 1 is chasing object 2 - pursuit dynamics detected"
            dynamic = "obj1_chasing"
        elif obj1_toward_obj2 < -0.3 and obj2_away_from_obj1 < -0.5:
            answer = "Object 1 is fleeing from object 2 - escape dynamics detected"
            dynamic = "obj1_fleeing"
        elif obj1_toward_obj2 > 0.5:
            answer = "Object 1 is approaching object 2 - may be chasing or intercepting"
            dynamic = "approaching"
        elif obj1_toward_obj2 < -0.5:
            answer = "Object 1 is moving away from object 2 - may be fleeing or diverging"
            dynamic = "diverging"
        else:
            answer = "No clear chase or flee pattern - objects moving independently"
            dynamic = "independent"
        
        return answer, {
            "dynamic": dynamic,
            "obj1_toward_obj2": obj1_toward_obj2,
            "obj2_away_from_obj1": obj2_away_from_obj1,
            "distance": distance
        }

    def _answer_cooperation_competition(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Assess whether objects are cooperating or competing."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "Insufficient objects to assess cooperation/competition", {"social_dynamic": "none"}
        
        velocities = computed_vel[active_mask]
        positions = end_states[active_mask, self.POSITION_IDX]
        
        vel_alignment = 0
        for i in range(count):
            for j in range(i + 1, count):
                if active_mask[i] and active_mask[j]:
                    speed_i = torch.norm(velocities[i]).item()
                    speed_j = torch.norm(velocities[j]).item()
                    if speed_i > 0.01 and speed_j > 0.01:
                        alignment = torch.dot(velocities[i] / speed_i, velocities[j] / speed_j).item()
                        vel_alignment += alignment
        
        num_pairs = count * (count - 1) / 2
        avg_alignment = vel_alignment / max(num_pairs, 1)
        
        center = positions.mean(dim=0)
        spreading = sum(torch.dot(velocities[i], positions[i] - center).item() 
                       for i in range(count) if active_mask[i]) / count
        
        if avg_alignment > 0.5:
            answer = "Cooperation detected - objects moving in coordinated fashion"
            dynamic = "cooperation"
        elif avg_alignment < -0.3 or spreading > 0.1:
            answer = "Competition or conflict - objects moving in opposing directions or spreading apart"
            dynamic = "competition"
        elif spreading < -0.1:
            answer = "Convergence - objects gathering together, possibly cooperative"
            dynamic = "convergence"
        else:
            answer = "Independent motion - no clear cooperation or competition"
            dynamic = "independent"
        
        return answer, {
            "social_dynamic": dynamic,
            "velocity_alignment": avg_alignment,
            "spreading": spreading
        }

    def _answer_causal_chain(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Explain the causal sequence leading to current state."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No objects to analyze", {"causal_events": []}
        
        causal_events = []
        
        for i in range(count):
            if not active_mask[i]:
                continue
            
            vel = computed_vel[i]
            speed = torch.norm(vel).item()
            
            if speed > 0.1:
                causal_events.append(f"Object {i + 1} received an impulse causing motion")
        
        if states.dim() == 3:
            for i in range(count):
                for j in range(i + 1, count):
                    if not (active_mask[i] and active_mask[j]):
                        continue
                    start_dist = torch.norm(start_states[i, self.POSITION_IDX] - start_states[j, self.POSITION_IDX]).item()
                    end_dist = torch.norm(end_states[i, self.POSITION_IDX] - end_states[j, self.POSITION_IDX]).item()
                    
                    radius_i = end_states[i, self.RADIUS_IDX].item() if end_states.shape[-1] > self.RADIUS_IDX else 0.5
                    radius_j = end_states[j, self.RADIUS_IDX].item() if end_states.shape[-1] > self.RADIUS_IDX else 0.5
                    contact_dist = radius_i + radius_j
                    
                    if start_dist > contact_dist and end_dist <= contact_dist * 1.5:
                        causal_events.append(f"Objects {i + 1} and {j + 1} likely collided, transferring momentum")
        
        if not causal_events:
            answer = "Objects appear to be in initial or steady state - no significant causal events detected"
        else:
            answer = "Causal sequence: " + "; ".join(causal_events)
        
        return answer, {"causal_events": causal_events}

    def _answer_future_prediction(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Predict what will happen next."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 1:
            return "No objects to predict", {"prediction": "none", "confidence": 0.0}
        
        predictions = []
        
        collision_imminent = False
        for i in range(count):
            for j in range(i + 1, count):
                if not (active_mask[i] and active_mask[j]):
                    continue
                
                pos_i = end_states[i, self.POSITION_IDX]
                pos_j = end_states[j, self.POSITION_IDX]
                vel_i = computed_vel[i]
                vel_j = computed_vel[j]
                
                relative_pos = pos_j - pos_i
                relative_vel = vel_j - vel_i
                distance = torch.norm(relative_pos).item()
                closing_speed = -torch.dot(relative_vel, relative_pos / (distance + 0.01)).item()
                
                radius_i = end_states[i, self.RADIUS_IDX].item() if end_states.shape[-1] > self.RADIUS_IDX else 0.5
                radius_j = end_states[j, self.RADIUS_IDX].item() if end_states.shape[-1] > self.RADIUS_IDX else 0.5
                
                if closing_speed > 0.05 and distance < (radius_i + radius_j) * 3:
                    collision_imminent = True
                    predictions.append(f"Objects {i + 1} and {j + 1} will collide")
        
        moving_count = 0
        for i in range(count):
            if active_mask[i] and torch.norm(computed_vel[i]).item() > 0.05:
                moving_count += 1
        
        if collision_imminent:
            answer = "Collision predicted: " + "; ".join(predictions)
            prediction_type = "collision"
            confidence = 0.8
        elif moving_count == 0:
            answer = "System appears stable - objects will remain stationary"
            prediction_type = "stable"
            confidence = 0.9
        else:
            answer = f"{moving_count} object(s) will continue along current trajectories"
            prediction_type = "continuation"
            confidence = 0.7
        
        return answer, {
            "prediction": prediction_type,
            "predictions": predictions,
            "confidence": confidence,
            "moving_count": moving_count
        }

    def _answer_counterfactual(self, states: torch.Tensor, mask: torch.Tensor) -> Tuple[str, Dict]:
        """Reason about what would happen if an object were removed."""
        end_states, start_states, computed_vel, active_mask, count = self._get_active_objects(states, mask)
        
        if count < 2:
            return "Insufficient objects for counterfactual reasoning", {"removed_object": None, "effect": "none"}
        
        removed_idx = 0
        
        removed_pos = end_states[removed_idx, self.POSITION_IDX]
        removed_vel = computed_vel[removed_idx]
        removed_speed = torch.norm(removed_vel).item()
        removed_mass = end_states[removed_idx, self.MASS_IDX].item() if end_states.shape[-1] > self.MASS_IDX else 1.0
        
        effects = []
        
        for i in range(count):
            if i == removed_idx or not active_mask[i]:
                continue
            
            other_pos = end_states[i, self.POSITION_IDX]
            other_vel = computed_vel[i]
            
            relative_pos = other_pos - removed_pos
            distance = torch.norm(relative_pos).item()
            
            if removed_speed > 0.05 and distance < 5.0:
                alignment = torch.dot(removed_vel / removed_speed, relative_pos / (distance + 0.01)).item()
                if alignment > 0.5:
                    effects.append(f"Object {i + 1} would not be impacted by object 1")
        
        if removed_speed > 0.1:
            momentum_removed = removed_speed * removed_mass
            effects.append(f"System momentum would decrease by {momentum_removed:.2f}")
        
        if not effects:
            answer = "Removing object 1 would have minimal effect on the system"
            effect_type = "minimal"
        else:
            answer = "Without object 1: " + "; ".join(effects)
            effect_type = "significant"
        
        return answer, {
            "removed_object": removed_idx,
            "effect": effect_type,
            "effects": effects
        }


def create_qa_dataset_from_physics(
    physics_dataset,
    num_samples: int = 1000,
    question_types: Optional[List[QuestionType]] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Create a QA dataset from a physics dataset.
    
    Args:
        physics_dataset: HDF5PhysicsDataset or similar
        num_samples: Number of QA pairs to generate
        question_types: Types of questions to include
        seed: Random seed
        
    Returns:
        List of dicts with states, mask, question, answer
    """
    generator = PhysicsQAGenerator(question_types=question_types, seed=seed)
    rng = random.Random(seed)
    
    qa_data = []
    dataset_size = len(physics_dataset)
    
    for i in range(num_samples):
        idx = rng.randint(0, dataset_size - 1)
        sample = physics_dataset[idx]
        
        states = sample['object_states']
        mask = sample['object_mask']
        
        qa_pair = generator.generate_qa_pair(states, mask)
        
        qa_data.append({
            'states': qa_pair.states,
            'mask': qa_pair.mask,
            'question': qa_pair.question,
            'answer': qa_pair.answer,
            'question_type': qa_pair.question_type.value,
            'metadata': qa_pair.metadata
        })
    
    return qa_data
