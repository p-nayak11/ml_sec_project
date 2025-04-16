import jax
import jax.numpy as jnp
from functools import partial
from jaxmarl.wrappers.baselines import JaxMARLWrapper

class ActionAttackWrapper(JaxMARLWrapper):
    """
    Wrapper to implement action attacks based on MARLSafe framework.
    Manipulates agent actions according to specified attack strategies.
    """
    
    def __init__(self, env, attack_config=None):
        super().__init__(env)
        self.attack_config = attack_config or {
            "enabled": False,
            "attack_type": "random",  # "random", "targeted", "inversion", "adversarial"
            "attack_probability": 0.5,  # Probability of performing attack at each step
            "attack_strength": 1.0,  # For partial attacks (0.0-1.0), how much to modify
            "targeted_agents": None,  # List of agent indices to attack, None means all
        }
        
        # Get discrete action space info
        self.action_spaces = {agent: self._env.action_space(agent) for agent in self._env.agents}
        self.num_actions = {agent: space.n for agent, space in self.action_spaces.items()}
        
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        # Just pass-through for reset
        return self._env.reset(key)
        
    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        attack_key, env_key = jax.random.split(key)
        
        if self.attack_config["enabled"]:
            # Apply the action attack
            attacked_action = self._attack_actions(attack_key, action)
            obs, env_state, reward, done, info = self._env.step(env_key, state, attacked_action)
            
            # Add attack information to info
            info = dict(info)
            info["action_attack"] = {
                "original_actions": action,
                "attacked_actions": attacked_action,
                "attack_applied": True
            }
        else:
            # No attack
            obs, env_state, reward, done, info = self._env.step(env_key, state, action)
            
        return obs, env_state, reward, done, info
    
    def _attack_actions(self, key, actions):
        """Apply attack to actions based on attack strategy"""
        
        # Skip attack with probability (1 - attack_probability)
        attack_prob_key, attack_key = jax.random.split(key)
        do_attack = jax.random.uniform(attack_prob_key) < self.attack_config["attack_probability"]
        
        def apply_attack(actions_dict):
            # Determine which agents to attack
            if self.attack_config["targeted_agents"] is None:
                agents_to_attack = self._env.agents
            else:
                agents_to_attack = [self._env.agents[i] for i in self.attack_config["targeted_agents"] 
                                   if i < len(self._env.agents)]
            
            # Create a new dictionary to avoid modifying the original
            attacked_actions = {k: v for k, v in actions_dict.items()}
            
            # Split the key for each agent
            agent_keys = jax.random.split(attack_key, len(agents_to_attack))
            
            for i, agent in enumerate(agents_to_attack):
                if agent not in self._env.agents:
                    continue
                
                agent_key = agent_keys[i]
                action = actions_dict[agent]
                num_actions = self.num_actions[agent]
                
                # Apply different attack strategies
                if self.attack_config["attack_type"] == "random":
                    # Completely random action
                    new_action = jax.random.randint(
                        agent_key, 
                        shape=action.shape, 
                        minval=0, 
                        maxval=num_actions
                    )
                    
                elif self.attack_config["attack_type"] == "targeted":
                    # Target specific action (for demonstration, we'll use action 0)
                    target_action = 0
                    new_action = jnp.zeros_like(action) + target_action
                    
                elif self.attack_config["attack_type"] == "inversion":
                    # Invert the action (for discrete actions, choose the "opposite")
                    # For simplicity in discrete spaces, we'll just add num_actions/2 and mod by num_actions
                    shift = num_actions // 2
                    new_action = (action + shift) % num_actions
                    
                elif self.attack_config["attack_type"] == "adversarial":
                    # In a real implementation, this would use a learned adversarial policy
                    # For simplicity, we'll implement a heuristic:
                    # Choose actions that are most different from the current one
                    options = jnp.arange(num_actions)
                    distances = jnp.abs(options - action)
                    # Pick the action with maximum distance from current
                    new_action = jnp.argmax(distances)
                    
                else:
                    # Default to no change if unknown attack type
                    new_action = action
                
                # Apply attack with specified strength (probabilistic for discrete actions)
                # For attack_strength=1.0, always use new_action
                # For attack_strength=0.0, always use original action
                # For values in-between, probabilistically choose
                apply_key, _ = jax.random.split(agent_key)
                do_modify = jax.random.uniform(apply_key) < self.attack_config["attack_strength"]
                attacked_actions[agent] = jnp.where(do_modify, new_action, action)
                
            return attacked_actions
        
        # Only apply attack with specified probability
        return jax.lax.cond(
            do_attack,
            lambda _: apply_attack(actions),
            lambda _: actions,
            operand=None
        )

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