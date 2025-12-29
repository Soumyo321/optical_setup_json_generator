import streamlit as st
import json
import re
import os
from datetime import datetime
# from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize model
model = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0
)

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are an expert optical simulation assistant that generates JSON configurations for optical setups.

## YOUR ROLE
- Parse user requests for optical components and layouts
- Generate valid JSON following the exact specification
- Modify existing JSON based on user change requests
- Always respond with ONLY valid JSON (no explanations, no markdown code blocks)

## JSON STRUCTURE RULES

### REQUIRED Root Structure
{
  "laser": {
    "id": "laser",
    "params": {
      "P": 1.0,
      "wavelength_nm": 650,
      "maxtem": 1
    }
  },
  "components": []
}

## COMPONENT TYPES & DEFAULTS

### 1. LASER (in components array)
{
  "id": "laser_1",
  "type": "laser",
  "params": {
    "P_W": 1.0,
    "wavelength_um": 0.65,
    "maxtem": 1,
    "f_Hz": 0,
    "Ph_rad": 0
  },
  "position": {"x": 100, "y": 260}
}

### 2. MIRROR
{
  "id": "mirror_1",
  "type": "mirror",
  "params": {
    "R": 0.95,
    "T": 0.05,
    "component_rotated": 0,
    "Rc_m": "inf"
  },
  "position": {"x": 350, "y": 260}
}

### 3. BEAM SPLITTER (BS)
{
  "id": "bs_1",
  "type": "bs",
  "params": {
    "R": 0.5,
    "T": 0.5,
    "component_rotated": 0,
    "Rc_m": "inf"
  },
  "position": {"x": 300, "y": 260}
}

### 4. LENS
{
  "id": "lens_1",
  "type": "lens",
  "params": {
    "f_um": 2000,
    "lensType": "convex"
  },
  "position": {"x": 400, "y": 260}
}

### 5. GLASS SLAB
{
  "id": "glassslab_1",
  "type": "glassslab",
  "params": {
    "thickness_um": 1000,
    "nr": 1.5
  },
  "position": {"x": 500, "y": 260}
}

### 6. DETECTOR
{
  "id": "detector_1",
  "type": "detector",
  "params": {
    "type": "photodiode"
  },
  "position": {"x": 600, "y": 260}
}

## POSITIONING RULES
- Default horizontal spacing: 250 pixels between components
- Base X: 100, Base Y: 260
- Canvas bounds: x(0-1000), y(0-600)
- "left" → x: 80-150
- "center" → x: 400-500
- "right" → x: 700-850

## ROTATION RULES
- component_rotated: ONLY 0, 90, 180, or 270
- User says "45 degrees" → component_rotated: 90
- User says "90 degrees" → component_rotated: 0 or 180

## CRITICAL RULES
1. ✅ ALWAYS include "laser" root object
2. ✅ ALWAYS include "components" array (can be empty)
3. ✅ Generate unique IDs: type_number (laser_1, mirror_2, etc.)
4. ✅ Keep R + T ≤ 1.0
5. ✅ Use ONLY valid component_rotated values: 0, 90, 180, 270
6. ❌ NEVER include: alpha_deg, debugAlpha, readings, totalPower
7. ❌ NEVER use component_rotated values like 45, 135, or any decimal values
8. ❌ NEVER use f_m for lens - use f_um instead
9. ❌ For lens params, use "lensType" not "type"

## OUTPUT FORMAT
- Respond with ONLY the JSON
- No markdown code blocks
- No explanations
- Just pure, valid JSON"""



def extract_json(text):
    """Extract JSON from response with better error handling"""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to fix common JSON issues
    try:
        # Remove any text before first {
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
            return json.loads(text)
    except:
        pass
    
    # Last resort: return a default valid JSON
    return {
        "laser": {
            "id": "laser",
            "params": {
                "P": 1.0,
                "wavelength_nm": 650,
                "maxtem": 1
            }
        },
        "components": []
    }


def validate_and_fix_json(json_data):
    """Validate and fix common JSON issues"""
    issues = []
    
    if "laser" not in json_data:
        return None, ["Missing 'laser' root object"]
    
    if "components" not in json_data:
        json_data["components"] = []
        issues.append("Added missing 'components' array")
    
    # Fix component IDs and parameters
    type_counters = {}
    
    for component in json_data.get("components", []):
        comp_type = component.get("type", "unknown")
        
        # Fix unique IDs
        type_counters[comp_type] = type_counters.get(comp_type, 0) + 1
        expected_id = f"{comp_type}_{type_counters[comp_type]}"
        
        if component.get("id") != expected_id:
            old_id = component.get("id", "none")
            component["id"] = expected_id
            issues.append(f"Fixed ID: {old_id} → {expected_id}")
        
        # Remove calculated fields
        if "params" in component:
            if "alpha_deg" in component["params"]:
                del component["params"]["alpha_deg"]
                issues.append(f"Removed alpha_deg from {component['id']}")
            
            # Fix rotation values
            if "component_rotated" in component["params"]:
                rotation = component["params"]["component_rotated"]
                if not isinstance(rotation, int) or rotation not in [0, 90, 180, 270]:
                    valid_rotations = [0, 90, 180, 270]
                    if isinstance(rotation, (int, float)):
                        normalized = rotation % 360
                        closest = min(valid_rotations, key=lambda x: abs(x - normalized))
                        component["params"]["component_rotated"] = closest
                        issues.append(f"Fixed rotation in {component['id']}: {rotation} → {closest}")
            
            # Fix lens parameters
            if comp_type == "lens":
                if "f_m" in component["params"]:
                    f_m = component["params"]["f_m"]
                    component["params"]["f_um"] = abs(f_m * 1_000_000)
                    del component["params"]["f_m"]
                    issues.append(f"Converted f_m to f_um in {component['id']}")
                
                if "type" in component["params"] and "lensType" not in component["params"]:
                    component["params"]["lensType"] = component["params"]["type"]
                    del component["params"]["type"]
                    issues.append(f"Renamed 'type' to 'lensType' in {component['id']}")
    
    return json_data, issues


def save_json(json_data, output_dir="output"):
    """Save JSON to file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optics_config_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return filepath


def generate_json(user_prompt, conversation_history, current_json):
    """Generate JSON from user prompt"""
    # Build conversation as list of dicts
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history
    for msg in conversation_history:
        messages.append(msg)
    
    # Add current request
    if current_json:
        context = f"\n\nCURRENT JSON:\n{json.dumps(current_json, indent=2)}\n\nModify based on: "
        messages.append({"role": "user", "content": context + user_prompt})
    else:
        messages.append({"role": "user", "content": user_prompt})
    
    try:
        # Invoke model
        response = model.invoke(messages)
        
        # Extract response content
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)
        
        # Extract JSON
        json_output = extract_json(response_content)
        
        # Validate and fix
        fixed_json, issues = validate_and_fix_json(json_output)
        
        if fixed_json is None:
            # Return default JSON if validation fails
            fixed_json = {
                "laser": {
                    "id": "laser",
                    "params": {
                        "P": 1.0,
                        "wavelength_nm": 650,
                        "maxtem": 1
                    }
                },
                "components": []
            }
            issues = ["Created default JSON due to validation errors"]
        
        filepath = save_json(fixed_json)
        
        return {
            "json": fixed_json,
            "issues": issues,
            "filepath": filepath,
            "response": response_content
        }
    
    except Exception as e:
        # Return default JSON on any error
        default_json = {
            "laser": {
                "id": "laser",
                "params": {
                    "P": 1.0,
                    "wavelength_nm": 650,
                    "maxtem": 1
                }
            },
            "components": []
        }
        filepath = save_json(default_json)
        
        return {
            "json": default_json,
            "issues": [f"Error occurred: {str(e)}. Created default JSON."],
            "filepath": filepath,
            "response": f"Error: {str(e)}"
        }



def main():
    st.set_page_config(
        page_title="Optics JSON Generator",
        page_icon="🔬",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_json" not in st.session_state:
        st.session_state.current_json = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.markdown('<div class="main-header">🔬 Optics Simulation JSON Generator</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Controls")
        
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.current_json = None
            st.session_state.chat_history = []
            st.success("Reset!")
            st.rerun()
        
        st.markdown("---")
        st.header("📖 Examples")
        
        examples = [
            "Add a red laser with a 45-degree mirror",
            "Create a Michelson interferometer",
            "Add a 50/50 beam splitter and two mirrors",
            "Add a convex lens with 2mm focal length",
            "Setup with laser, lens, and detector"
        ]
        
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.pending_input = ex
        
        st.markdown("---")
        st.header("ℹ️ Components")
        st.markdown("""
        - **laser**: Light source
        - **mirror**: Reflective surface
        - **bs**: Beam splitter
        - **lens**: Focusing element
        - **glassslab**: Glass block
        - **detector**: Sensor
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.chat_message("user").write(chat["content"])
            else:
                st.chat_message("assistant").write(chat["content"])
        
        # Handle pending input from sidebar
        if "pending_input" in st.session_state:
            user_input = st.session_state.pending_input
            del st.session_state.pending_input
        else:
            user_input = st.chat_input("Describe your optical setup...")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("Generating..."):
                try:
                    result = generate_json(
                        user_input,
                        st.session_state.conversation_history,
                        st.session_state.current_json
                    )
                    
                    st.session_state.current_json = result["json"]
                    st.session_state.conversation_history.append({"role": "user", "content": user_input})
                    st.session_state.conversation_history.append({"role": "assistant", "content": result["response"]})
                    
                    msg = f"✅ Generated! Saved to: {result['filepath']}"
                    if result["issues"]:
                        msg += f"\n\n🔧 Fixed/Handled {len(result['issues'])} issue(s):\n"
                        msg += "\n".join(f"- {i}" for i in result["issues"])
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": msg})
                    
                except Exception as e:
                    # Fallback: create default JSON
                    default_json = {
                        "laser": {
                            "id": "laser",
                            "params": {"P": 1.0, "wavelength_nm": 650, "maxtem": 1}
                        },
                        "components": []
                    }
                    st.session_state.current_json = default_json
                    filepath = save_json(default_json)
                    
                    error_msg = f"⚠️ Error: {str(e)}\n\nCreated default JSON instead. Saved to: {filepath}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
    
    with col2:
        st.header("📄 Generated JSON")
        
        if st.session_state.current_json:
            num_components = len(st.session_state.current_json.get("components", []))
            st.metric("Components", num_components)
            
            json_str = json.dumps(st.session_state.current_json, indent=2)
            st.code(json_str, language="json")
            
            st.download_button(
                "📥 Download",
                json_str,
                "optics_config.json",
                "application/json",
                use_container_width=True
            )
            
            with st.expander("🔍 Details"):
                for comp in st.session_state.current_json.get("components", []):
                    st.markdown(f"**{comp.get('id')}** ({comp.get('type')})")
                    st.json(comp.get("params", {}))
        else:
            st.info("👈 Describe your optical setup to generate JSON")


if __name__ == "__main__":
    main()