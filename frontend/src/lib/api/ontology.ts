import apiClient from './client'

export interface TaxonomyItem {
  value: string
  label: string
}

export const ontologyApi = {
  systems: async () => {
    const response = await apiClient.get<TaxonomyItem[]>('/ontology/systems')
    return response.data
  },
  specialties: async () => {
    const response = await apiClient.get<TaxonomyItem[]>('/ontology/specialties')
    return response.data
  },
}

export default ontologyApi
