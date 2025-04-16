import jax
import jax.numpy as jnp
from functools import partial
from jaxmarl.wrappers.baselines import JaxMARLWrapper

class StateAttackWrapper(JaxMARLWrapper):
    """
    Wrapper to implement state attacks based on MARLSafe framework.
    Applies perturbations to observations according to specified attack strategies.
    """
    
    def __init__(self, env, attack_config=None):
        super().__init__(env)
        self.attack_config = attack_config or {
            "enabled": False,
            "attack_type": "random",  # "random", "targeted", "strategic"
            "attack_magnitude": 0.1,  # Maximum magnitude of perturbation
            "attack_probability": 0.5,  # Probability of performing attack at each step
            "targeted_agents": None,  # List of agent indices to attack, None means all
            "attack_features": None,  # List of feature indices to attack, None means all
        }
        
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        attack_key, key = jax.random.split(key)
        
        if self.attack_config["enabled"]:
            obs = self._attack_observation(attack_key, obs)
            
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        
        attack_key, key = jax.random.split(key)
        if self.attack_config["enabled"]:
            obs = self._attack_observation(attack_key, obs)
            
        return obs, env_state, reward, done, info
    
    def _attack_observation(self, key, obs):
        """Apply perturbation to observations based on attack strategy"""
        
        # Skip attack with probability (1 - attack_probability)
        attack_prob_key, perturbation_key, feature_key, agent_key = jax.random.split(key, 4)
        do_attack = jax.random.uniform(attack_prob_key) < self.attack_config["attack_probability"]
        
        def apply_attack(obs_dict):
            # Determine which agents to attack
            agents_to_attack = self.attack_config["targeted_agents"] or self._env.agents
            
            # Create a new dictionary to avoid modifying the original
            attacked_obs = {k: v for k, v in obs_dict.items()}
            
            for agent in agents_to_attack:
                if agent not in self._env.agents:
                    continue
                
                # Skip non-array observations (like "world_state" which is handled separately)
                if not isinstance(obs_dict[agent], jnp.ndarray):
                    continue
                    
                features = self.attack_config["attack_features"]
                if features is None:
                    features = list(range(obs_dict[agent].shape[-1]))
                
                # Create perturbation based on attack type
                if self.attack_config["attack_type"] == "random":
                    # Random uniform noise in [-magnitude, magnitude]
                    perturbation = jax.random.uniform(
                        perturbation_key, 
                        obs_dict[agent].shape,
                        minval=-self.attack_config["attack_magnitude"],
                        maxval=self.attack_config["attack_magnitude"]
                    )
                elif self.attack_config["attack_type"] == "targeted":
                    # Target specific directions (always positive perturbation)
                    perturbation = jnp.ones_like(obs_dict[agent]) * self.attack_config["attack_magnitude"]
                elif self.attack_config["attack_type"] == "strategic":
                    # More sophisticated attack (could implement adversarial attacks here)
                    # For now we'll use a mix of positive and negative perturbations
                    signs = jax.random.choice(
                        perturbation_key, 
                        jnp.array([-1.0, 1.0]), 
                        shape=obs_dict[agent].shape
                    )
                    perturbation = signs * self.attack_config["attack_magnitude"]
                else:
                    # Default to zero perturbation if unknown attack type
                    perturbation = jnp.zeros_like(obs_dict[agent])
                
                # Only perturb selected features
                feature_mask = jnp.zeros_like(obs_dict[agent], dtype=bool)
                for f in features:
                    if f < feature_mask.shape[-1]:
                        feature_mask = feature_mask.at[..., f].set(True)
                
                # Apply perturbation using mask
                attacked_obs[agent] = jnp.where(
                    feature_mask,
                    obs_dict[agent] + perturbation,
                    obs_dict[agent]
                )
                
            # Handle world_state if it exists
            if "world_state" in obs_dict:
                # Apply similar perturbation to world_state
                if self.attack_config["attack_type"] == "random":
                    world_perturbation = jax.random.uniform(
                        perturbation_key, 
                        obs_dict["world_state"].shape,
                        minval=-self.attack_config["attack_magnitude"],
                        maxval=self.attack_config["attack_magnitude"]
                    )
                else:
                    # Skip world_state for other attack types for simplicity
                    world_perturbation = jnp.zeros_like(obs_dict["world_state"])
                
                attacked_obs["world_state"] = obs_dict["world_state"] + world_perturbation
                
            return attacked_obs
        
        # Only apply attack with specified probability
        return jax.lax.cond(
            do_attack,
            lambda _: apply_attack(obs),
            lambda _: obs,
            operand=None
        )