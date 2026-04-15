"""State management for Web UI workflows.

Provides centralized state management using Streamlit session_state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import streamlit as st


@dataclass
class JointWorkflowState:
    """State for joint workflow (Flux + LTX I2V)."""

    flux_image: Optional[str] = None
    flux_prompt: str = ""
    scene_keywords: str = ""
    motion_description: str = ""
    i2v_prompt: Optional[str] = None
    video_path: Optional[str] = None
    step: Optional[str] = None

    # Pipeline configuration
    use_two_stage: bool = True  # Default to two-stage
    use_hq: bool = True  # Default to HQ


@dataclass
class FlashWorkflowState:
    """State for flash workflow (multi-segment video)."""

    theme_words: str = ""
    scene_count: int = 5
    frames_per_segment: int = 49
    seed: int = 42
    scenes: List[str] = field(default_factory=list)
    segments: List[str] = field(default_factory=list)
    status: str = "idle"
    current_segment: int = 0
    output_path: Optional[str] = None
    temp_dir: Optional[str] = None

    # Mode selection
    mode: str = "i2v"  # "i2v" or "t2v"
    use_two_stage: bool = True  # Default to two-stage
    use_hq: bool = True  # Default to HQ


@dataclass
class I2VWorkflowState:
    """State for I2V workflow (image upload + LTX I2V)."""

    uploaded_image: Optional[str] = None
    prompt: str = ""
    video_path: Optional[str] = None
    step: Optional[str] = None

    # Pipeline configuration
    use_two_stage: bool = True  # Default to two-stage
    use_hq: bool = True  # Default to HQ


class StateManager:
    """Centralized state manager using Streamlit session_state."""

    @staticmethod
    def get_joint_workflow() -> JointWorkflowState:
        """Get or create joint workflow state."""
        if "joint_workflow" not in st.session_state:
            st.session_state.joint_workflow = JointWorkflowState()
        return st.session_state.joint_workflow

    @staticmethod
    def get_flash_workflow() -> FlashWorkflowState:
        """Get or create flash workflow state."""
        if "flash_workflow" not in st.session_state:
            st.session_state.flash_workflow = FlashWorkflowState()
        return st.session_state.flash_workflow

    @staticmethod
    def get_i2v_workflow() -> I2VWorkflowState:
        """Get or create I2V workflow state."""
        if "i2v_workflow" not in st.session_state:
            st.session_state.i2v_workflow = I2VWorkflowState()
        return st.session_state.i2v_workflow

    @staticmethod
    def reset_joint_workflow():
        """Reset joint workflow state."""
        st.session_state.joint_workflow = JointWorkflowState()

    @staticmethod
    def reset_flash_workflow():
        """Reset flash workflow state."""
        st.session_state.flash_workflow = FlashWorkflowState()

    @staticmethod
    def reset_i2v_workflow():
        """Reset I2V workflow state."""
        st.session_state.i2v_workflow = I2VWorkflowState()

    @staticmethod
    def update_joint_workflow(**kwargs):
        """Update joint workflow state fields."""
        state = StateManager.get_joint_workflow()
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

    @staticmethod
    def update_flash_workflow(**kwargs):
        """Update flash workflow state fields."""
        state = StateManager.get_flash_workflow()
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

    @staticmethod
    def update_i2v_workflow(**kwargs):
        """Update I2V workflow state fields."""
        state = StateManager.get_i2v_workflow()
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
