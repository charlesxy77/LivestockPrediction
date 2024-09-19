import React, { useState, useEffect } from 'react';
import axios from 'axios';

const App = () => {
  const [inputs, setInputs] = useState({
    DM: '', CP: '', CF: '', NDF: '', ADF: '', ADL: '', ASH: ''
  });
  const [outputs, setOutputs] = useState(null);
  const [error, setError] = useState(null);

  const parameterNames = {
    DM: 'Dry Matter',
    CP: 'Crude Protein',
    CF: 'Crude Fiber',
    NDF: 'Neutral Detergent Fiber',
    ADF: 'Acid Detergent Fiber',
    ADL: 'Acid Detergent Lignin',
    ASH: 'Ash'
  };

  useEffect(() => {
    axios.get('http://localhost:5000/test')
      .then(response => console.log('Backend connection successful:', response.data))
      .catch(error => console.error('Backend connection failed:', error));
  }, []);

  const handleInputChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/predict', inputs);
      setOutputs(response.data);
    } catch (error) {
      console.error('Error:', error);
      setError(error.response?.data?.error || error.message || 'An error occurred');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 sm:p-6 md:p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl md:text-4xl text-blue-600 font-bold mb-8 text-center">Livestock Feed Predictor</h1>

        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="p-6 md:p-8">
            <h2 className="text-2xl font-semibold mb-6 text-gray-800">Input Parameters</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              {Object.keys(inputs).map((key) => (
                <div key={key} className="flex flex-col">
                  <label htmlFor={key} className="text-sm font-medium text-gray-700 mb-1">{parameterNames[key]} ({key})</label>
                  <input
                    type="number"
                    id={key}
                    name={key}
                    value={inputs[key]}
                    onChange={handleInputChange}
                    className="w-full p-2 rounded border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    placeholder={`Enter ${key}`}
                  />
                </div>
              ))}
              <button
                type="submit"
                className="w-full py-2 px-4 bg-blue-600 text-white rounded hover:bg-blue-700 transition duration-300 flex items-center justify-center text-lg font-semibold"
              >
                Predict
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </form>

            {outputs && (
              <div className="mt-10">
                <h2 className="text-2xl font-semibold mb-6 text-gray-800">Prediction Results</h2>
                <div className="grid grid-cols-2 gap-4 mb-6">
                  {Object.entries(outputs).map(([key, value]) => (
                    <div key={key} className="p-4 rounded-lg bg-blue-50 border border-blue-200">
                      <h3 className="text-lg font-medium text-blue-800">{key}</h3>
                      <p className="text-2xl font-bold text-blue-600">{value.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
                <div className="p-6 rounded-lg bg-green-50 border border-green-200">
                  <h3 className="text-xl font-semibold mb-3 text-green-800">Key Insights:</h3>
                  <ul className="list-disc pl-5 space-y-2 text-green-700">
                    <li>DMD is {outputs.DMD < 50 ? 'lower' : 'higher'} than average, indicating {outputs.DMD < 50 ? 'poor' : 'good'} digestibility.</li>
                    <li>OMD suggests {outputs.OMD < 60 ? 'low' : 'high'} organic matter degradation.</li>
                    <li>ME value indicates {outputs.ME < 10 ? 'low' : 'high'} energy content in the feed.</li>
                    <li>CH4 production is {outputs.CH4 < 25 ? 'relatively low' : 'significant'}, consider environmental impact.</li>
                  </ul>
                </div>
              </div>
            )}

            {error && (
              <div className="mt-8 p-4 rounded-lg bg-red-100 border border-red-300 text-red-700">
                Error: {error}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;