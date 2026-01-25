import React from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stage } from '@react-three/drei'

export default function Viz({ filename }) {
    return (
        <div className="w-full h-full relative bg-gray-900">
            <Canvas shadows dpr={[1, 2]} camera={{ position: [5, 5, 5], up: [0, 0, 1] }}>
                <OrbitControls makeDefault up={[0, 0, 1]} />
                <Stage environment="city">
                    <mesh>
                        <boxGeometry />
                        <meshStandardMaterial color="orange" />
                    </mesh>
                </Stage>
            </Canvas>
            <div className="absolute top-4 left-4 text-white text-xs">
                {filename || 'No File'}
            </div>
        </div>
    )
}
