import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

// API base URL - Using relative path for internal Docker network communication
const API_BASE_URL = '/api';

function App() {
  const [activeTab, setActiveTab] = useState('crop');
  const [loading, setLoading] = useState(false);
  const [cropParams, setCropParams] = useState({
    Nitrogen: 90,
    Phosphorus: 42,
    Potassium: 43,
    Temperature: 20.87,
    Humidity: 82.00,
    pH_Value: 6.5,
    Rainfall: 202.93
  });
  const [cropResults, setCropResults] = useState(null);
  
  const [selectedState, setSelectedState] = useState('');
  const [states, setStates] = useState([]);
  const [yieldResults, setYieldResults] = useState(null);
  
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  
  // Fetch available states when component mounts
  useEffect(() => {
    axios.get(`${API_BASE_URL}/info`)
      .then(response => {
        if (response.data.available_states) {
          setStates(response.data.available_states);
          if (response.data.available_states.length > 0) {
            setSelectedState(response.data.available_states[0]);
          }
        }
      })
      .catch(error => {
        console.error("Error fetching states:", error);
      });
  }, []);

  const handleCropParamChange = (e) => {
    setCropParams({
      ...cropParams,
      [e.target.name]: parseFloat(e.target.value)
    });
  };

  const handleCropSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/recommend-crop`, cropParams);
      setCropResults(response.data);
    } catch (error) {
      console.error("Error recommending crops:", error);
      setCropResults({ success: false, message: "Failed to get crop recommendations" });
    }
    setLoading(false);
  };

  const handleYieldSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/predict-yield`, { state: selectedState });
      setYieldResults(response.data);
    } catch (error) {
      console.error("Error predicting yield:", error);
      setYieldResults({ success: false, message: "Failed to get yield prediction" });
    }
    setLoading(false);
  };

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/ask`, { question });
      setAnswer(response.data);
    } catch (error) {
      console.error("Error asking question:", error);
      setAnswer({ success: false, message: "Failed to get an answer" });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="bg-green-700 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">Agricultural Insights</h1>
          <p className="text-green-100">Intelligent Crop and Yield Analysis with scikit-learn</p>
        </div>
      </header>
      
      <nav className="bg-green-800 text-white">
        <div className="container mx-auto px-4">
          <div className="flex space-x-1">
            <button 
              className={`px-4 py-3 hover:bg-green-600 transition ${activeTab === 'crop' ? 'bg-green-600 font-medium' : ''}`}
              onClick={() => setActiveTab('crop')}
            >
              Crop Recommendation
            </button>
            <button 
              className={`px-4 py-3 hover:bg-green-600 transition ${activeTab === 'yield' ? 'bg-green-600 font-medium' : ''}`}
              onClick={() => setActiveTab('yield')}
            >
              Yield Prediction
            </button>
            <button 
              className={`px-4 py-3 hover:bg-green-600 transition ${activeTab === 'qa' ? 'bg-green-600 font-medium' : ''}`}
              onClick={() => setActiveTab('qa')}
            >
              Ask a Question
            </button>
          </div>
        </div>
      </nav>
      
      <main className="container mx-auto px-4 py-8">
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-xl">
              <p className="text-xl">Loading...</p>
            </div>
          </div>
        )}
        
        {activeTab === 'crop' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold text-green-800 mb-6">Crop Recommendation</h2>
            
            <form onSubmit={handleCropSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block mb-2 text-gray-700">Nitrogen (N):</label>
                <input
                  type="number"
                  name="Nitrogen"
                  value={cropParams.Nitrogen}
                  onChange={handleCropParamChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  step="any"
                  required
                />
              </div>
              
              <div>
                <label className="block mb-2 text-gray-700">Phosphorus (P):</label>
                <input
                  type="number"
                  name="Phosphorus"
                  value={cropParams.Phosphorus}
                  onChange={handleCropParamChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  step="any"
                  required
                />
              </div>
              
              <div>
                <label className="block mb-2 text-gray-700">Potassium (K):</label>
                <input
                  type="number"
                  name="Potassium"
                  value={cropParams.Potassium}
                  onChange={handleCropParamChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  step="any"
                  required
                />
              </div>
              
              <div>
                <label className="block mb-2 text-gray-700">Temperature (°C):</label>
                <input
                  type="number"
                  name="Temperature"
                  value={cropParams.Temperature}
                  onChange={handleCropParamChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  step="any"
                  required
                />
              </div>
              
              <div>
                <label className="block mb-2 text-gray-700">Humidity (%):</label>
                <input
                  type="number"
                  name="Humidity"
                  value={cropParams.Humidity}
                  onChange={handleCropParamChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  step="any"
                  required
                />
              </div>
              
              <div>
                <label className="block mb-2 text-gray-700">pH Value:</label>
                <input
                  type="number"
                  name="pH_Value"
                  value={cropParams.pH_Value}
                  onChange={handleCropParamChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  step="any"
                  required
                />
              </div>
              
              <div>
                <label className="block mb-2 text-gray-700">Rainfall (mm):</label>
                <input
                  type="number"
                  name="Rainfall"
                  value={cropParams.Rainfall}
                  onChange={handleCropParamChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  step="any"
                  required
                />
              </div>
              
              <div className="md:col-span-2">
                <button
                  type="submit"
                  className="px-6 py-3 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 transition"
                >
                  Get Recommendations
                </button>
              </div>
            </form>
            
            {cropResults && cropResults.success && (
              <div className="mt-8">
                <h3 className="text-xl font-semibold text-green-800 mb-4">Recommended Crops</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {cropResults.recommendations.map((crop, index) => (
                    <div key={index} className="bg-green-50 border border-green-200 rounded-md p-4">
                      <h4 className="font-medium text-lg">{crop.crop}</h4>
                      <p className="text-sm text-gray-600">Confidence: {(crop.probability * 100).toFixed(2)}%</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {cropResults && !cropResults.success && (
              <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-md text-red-700">
                {cropResults.message}
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'yield' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold text-green-800 mb-6">Rice Yield Prediction</h2>
            
            <form onSubmit={handleYieldSubmit} className="mb-6">
              <div className="mb-4">
                <label className="block mb-2 text-gray-700">Select State:</label>
                <select
                  value={selectedState}
                  onChange={(e) => setSelectedState(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  required
                >
                  <option value="">-- Select a state --</option>
                  {states.map(state => (
                    <option key={state} value={state}>{state}</option>
                  ))}
                </select>
              </div>
              
              <button
                type="submit"
                className="px-6 py-3 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 transition"
              >
                Predict Yield
              </button>
            </form>
            
            {yieldResults && yieldResults.success && (
              <div className="mt-8">
                <h3 className="text-xl font-semibold text-green-800 mb-4">Yield Analysis for {yieldResults.state}</h3>
                <div className="bg-green-50 border border-green-200 rounded-md p-4">
                  <h4 className="font-medium text-lg mb-2">Important Factors</h4>
                  
                  <div className="space-y-3">
                    {Object.entries(yieldResults.important_factors).map(([feature, data]) => (
                      <div key={feature}>
                        <div className="flex justify-between items-center text-sm">
                          <span>{feature}</span>
                          <span>{(data.importance * 100).toFixed(2)}% importance</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-600 h-2 rounded-full"
                            style={{ width: `${Math.min(data.importance * 100, 100)}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            {yieldResults && !yieldResults.success && (
              <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-md text-red-700">
                {yieldResults.message}
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'qa' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold text-green-800 mb-6">Ask a Question</h2>
            
            <form onSubmit={handleQuestionSubmit} className="mb-6">
              <div className="mb-4">
                <label className="block mb-2 text-gray-700">Your Question:</label>
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  rows="3"
                  placeholder="E.g., What crop should I grow if I have nitrogen 80, phosphorus 40, potassium 40, temperature 25, humidity 70, pH 6.5, and rainfall 200?"
                  required
                ></textarea>
              </div>
              
              <button
                type="submit"
                className="px-6 py-3 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 transition"
              >
                Ask Question
              </button>
            </form>
            
            {answer && (
              <div className="mt-8">
                <h3 className="text-xl font-semibold text-green-800 mb-4">Answer</h3>
                <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                  {answer.success && answer.question_type === 'crop_recommendation' && answer.recommendations && (
                    <div>
                      <h4 className="font-medium text-lg mb-2">Recommended Crops:</h4>
                      <ul className="list-disc pl-5 space-y-1">
                        {answer.recommendations.map((crop, index) => (
                          <li key={index}>{crop.crop} ({(crop.probability * 100).toFixed(2)}% confidence)</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {answer.success && answer.question_type === 'yield_prediction' && (
                    <div>
                      <h4 className="font-medium text-lg mb-2">Yield Information for {answer.state}</h4>
                      <p>The most important factors affecting rice yield in this region are:</p>
                      <ul className="list-disc pl-5 space-y-1">
                        {Object.entries(answer.important_factors).map(([feature, data], index) => (
                          <li key={index}>{feature} ({(data.importance * 100).toFixed(2)}% importance)</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {answer.success && answer.question_type === 'general_info' && (
                    <div>
                      <h4 className="font-medium text-lg mb-2">General Information</h4>
                      <p>We have information about {answer.available_crops?.length || 0} different crops and {answer.available_states?.length || 0} states.</p>
                    </div>
                  )}
                  
                  {!answer.success && (
                    <p className="text-red-700">{answer.message}</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
      
      <footer className="bg-green-900 text-white py-6">
        <div class="container mx-auto px-4 text-center">
          <p>Agricultural Insights &copy; 2025</p>
          <p class="text-sm text-green-300 mt-1">Powered by scikit-learn and React</p>
        </div>
      </footer>
    </div>
  )
}

export default App
