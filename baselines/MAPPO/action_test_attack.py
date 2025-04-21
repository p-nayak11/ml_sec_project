import jax
import jax.numpy as jnp

def apply_action_test_attack(actions, env, traitor_idx=0):
    """
    Apply action test attack by designating one agent as a traitor
    that takes actions to harm team performance.
    
    Args:
        actions: Dictionary of actions for each agent
        env: The environment
        traitor_idx: Index of the agent to act as traitor (default: 0)
        
    Returns:
        Dictionary of actions with traitor's actions modified
    """
    # Create a copy of the actions dictionary
    modified_actions = dict(actions)
    
    # Get the traitor agent ID
    traitor_agent_id = env.agents[traitor_idx]
    
    action = actions[traitor_agent_id]
    action_dim = env.action_space(traitor_agent_id).n
    
    adversarial_action = (action + action_dim // 2) % action_dim
    
    # Replace the traitor's action
    modified_actions[traitor_agent_id] = adversarial_action
    
    return modified_actions

def vmap_apply_action_test_attack(env_acts, env, traitor_idx=0):
    """Vectorized version of apply_action_test_attack for use with jax.vmap"""
    return jax.vmap(lambda act: apply_action_test_attack(act, env, traitor_idx))(env_acts)