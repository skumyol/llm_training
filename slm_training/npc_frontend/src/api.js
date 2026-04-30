import axios from 'axios'

const BASE = '/api'

export const api = {
  health:       ()                    => axios.get(`${BASE}/health`),
  models:       ()                    => axios.get(`${BASE}/models`),
  catalog:      ()                    => axios.get(`${BASE}/models/catalog`),
  selectModel:  (stage, model_id)     => axios.post(`${BASE}/models/select`, { stage, model_id }),
  evalSummary:  ()                    => axios.get(`${BASE}/eval`),

  listNpcs:     ()                    => axios.get(`${BASE}/npcs`),
  registerNpc:  (npc_id, profile_text) =>
    axios.post(`${BASE}/npcs`, { npc_id, profile_text }),
  removeNpc:    (npc_id)              => axios.delete(`${BASE}/npcs/${npc_id}`),
  npcState:     (npc_id)              => axios.get(`${BASE}/npcs/${npc_id}/state`),

  chat:         (npc_id, message)     =>
    axios.post(`${BASE}/chat/${npc_id}`, { message }),
  reset:        (npc_id)              => axios.post(`${BASE}/reset/${npc_id}`),

  loadWorld:    (yaml_path)           =>
    axios.post(`${BASE}/load-world`, { yaml_path }),
}
