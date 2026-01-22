import React from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stage } from '@react-three/drei'

export default function Viz({ filename }) {
    return (
        <div className="w-full h-full relative bg-gray-900">
            <Canvas shadows dpr={[1, 2]}>
                <OrbitControls makeDefault />
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
