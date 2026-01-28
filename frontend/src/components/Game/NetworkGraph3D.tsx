import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Line, Html } from '@react-three/drei'
import * as THREE from 'three'
import { motion } from 'framer-motion'
import { GameState, Node, Connection } from '@/types'

interface NetworkGraph3DProps {
  gameState: GameState
  onNodeClick?: (nodeId: string) => void
  selectedNode?: string | null
}

interface NodeMeshProps {
  node: Node
  position: [number, number, number]
  isSelected: boolean
  isHovered: boolean
  onClick: () => void
  onHover: (hovered: boolean) => void
}

interface ConnectionLineProps {
  start: [number, number, number]
  end: [number, number, number]
  connection: Connection
}

// Get node color based on status and control
function getNodeColor(node: Node): string {
  if (node.isCompromised) return '#ff0040' // Red for compromised
  if (node.controlledBy === 'defender') return '#00ff88' // Cyber primary for defender
  if (node.controlledBy === 'attacker') return '#ff6b00' // Orange for attacker
  if (node.isHoneypot) return '#8b5cf6' // Purple for honeypot
  return '#0ea5e9' // Cyan for neutral
}

// Calculate 3D positions for nodes using force-directed layout
function calculateNodePositions(nodes: Node[]): Map<string, [number, number, number]> {
  const positions = new Map<string, [number, number, number]>()
  
  // Simple spherical layout based on node criticality and type
  const nodeArray = Object.values(nodes)
  const radius = 5
  
  nodeArray.forEach((node, index) => {
    const angle = (index / nodeArray.length) * Math.PI * 2
    const heightVariation = (node.criticality / 10) * 2 - 1
    const radiusVariation = radius * (0.7 + Math.random() * 0.6)
    
    const x = Math.cos(angle) * radiusVariation
    const y = heightVariation + (Math.random() - 0.5)
    const z = Math.sin(angle) * radiusVariation
    
    positions.set(node.id, [x, y, z])
  })
  
  return positions
}

// Individual node mesh component
function NodeMesh({ node, position, isSelected, isHovered, onClick, onHover }: NodeMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  
  const color = getNodeColor(node)
  const size = 0.3 + (node.criticality / 10) * 0.3
  
  useFrame((state) => {
    if (meshRef.current) {
      // Subtle floating animation
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.05
      
      // Pulse effect for selected/hovered
      if (isSelected || isHovered) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 4) * 0.1
        meshRef.current.scale.setScalar(scale)
      } else {
        meshRef.current.scale.setScalar(1)
      }
    }
    
    if (glowRef.current) {
      glowRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.05
      const glowScale = isSelected || isHovered ? 2.5 : 1.8
      glowRef.current.scale.setScalar(glowScale)
    }
  })
  
  return (
    <group position={position}>
      {/* Glow effect */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[size * 1.5, 16, 16]} />
        <meshBasicMaterial 
          color={color} 
          transparent 
          opacity={isSelected ? 0.3 : isHovered ? 0.2 : 0.1} 
        />
      </mesh>
      
      {/* Main node */}
      <mesh 
        ref={meshRef}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
        onPointerOver={(e) => { e.stopPropagation(); onHover(true); }}
        onPointerOut={() => onHover(false)}
      >
        <dodecahedronGeometry args={[size, 0]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color}
          emissiveIntensity={isSelected ? 0.8 : isHovered ? 0.5 : 0.3}
          metalness={0.7}
          roughness={0.3}
        />
      </mesh>
      
      {/* Node label */}
      {(isHovered || isSelected) && (
        <Html distanceFactor={10} position={[0, size + 0.5, 0]}>
          <div className="bg-cyber-dark/90 border border-cyber-primary/50 rounded px-2 py-1 text-xs font-mono whitespace-nowrap">
            <div className="text-cyber-primary font-bold">{node.name}</div>
            <div className="text-gray-400 text-[10px]">{node.type}</div>
            <div className="flex gap-2 text-[10px]">
              <span className="text-cyan-400">HP: {node.health}</span>
              <span className="text-yellow-400">Crit: {node.criticality}</span>
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

// Connection line component
function ConnectionLine({ start, end, connection }: ConnectionLineProps) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end)
  ], [start, end])
  
  const color = connection.isEncrypted ? '#00ff88' : '#0ea5e9'
  const opacity = connection.bandwidth > 50 ? 0.8 : 0.4
  
  return (
    <Line
      points={points}
      color={color}
      lineWidth={1 + connection.bandwidth / 50}
      transparent
      opacity={opacity}
    />
  )
}

// Animated background particles
function BackgroundParticles() {
  const particlesRef = useRef<THREE.Points>(null)
  const count = 200
  
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 30
      pos[i * 3 + 1] = (Math.random() - 0.5) * 30
      pos[i * 3 + 2] = (Math.random() - 0.5) * 30
    }
    return pos
  }, [])
  
  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.02
      particlesRef.current.rotation.x = state.clock.elapsedTime * 0.01
    }
  })
  
  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial size={0.05} color="#00ff88" transparent opacity={0.5} />
    </points>
  )
}

// Camera controls wrapper
function CameraController() {
  const { camera } = useThree()
  
  useFrame(() => {
    // Keep camera looking at center
    camera.lookAt(0, 0, 0)
  })
  
  return (
    <OrbitControls 
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      minDistance={5}
      maxDistance={25}
      autoRotate={false}
      autoRotateSpeed={0.5}
    />
  )
}

// Main scene component
function NetworkScene({ gameState, onNodeClick, selectedNode }: NetworkGraph3DProps) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  
  const nodePositions = useMemo(() => {
    if (!gameState?.network?.nodes) return new Map()
    return calculateNodePositions(
      Object.values(gameState.network.nodes)
    )
  }, [gameState?.network])
  
  if (!gameState?.network) {
    return null
  }
  
  const nodes = Object.values(gameState.network.nodes)
  const connections = gameState.network.connections
  
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#00ff88" />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#0ea5e9" />
      <pointLight position={[0, 10, 0]} intensity={0.3} color="#8b5cf6" />
      
      {/* Background */}
      <BackgroundParticles />
      
      {/* Connections */}
      {connections.map((conn, index) => {
        const startPos = nodePositions.get(conn.sourceId)
        const endPos = nodePositions.get(conn.targetId)
        if (!startPos || !endPos) return null
        
        return (
          <ConnectionLine
            key={`${conn.sourceId}-${conn.targetId}-${index}`}
            start={startPos}
            end={endPos}
            connection={conn}
          />
        )
      })}
      
      {/* Nodes */}
      {nodes.map((node) => {
        const position = nodePositions.get(node.id)
        if (!position) return null
        
        return (
          <NodeMesh
            key={node.id}
            node={node}
            position={position}
            isSelected={selectedNode === node.id}
            isHovered={hoveredNode === node.id}
            onClick={() => onNodeClick?.(node.id)}
            onHover={(hovered) => setHoveredNode(hovered ? node.id : null)}
          />
        )
      })}
      
      {/* Camera controls */}
      <CameraController />
    </>
  )
}

// Main exported component
function NetworkGraph3D({ gameState, onNodeClick, selectedNode }: NetworkGraph3DProps) {
  return (
    <motion.div 
      className="w-full h-full rounded-xl overflow-hidden border border-cyber-primary/30"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Canvas
        camera={{ position: [0, 5, 15], fov: 60 }}
        style={{ background: 'linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%)' }}
      >
        <NetworkScene 
          gameState={gameState} 
          onNodeClick={onNodeClick}
          selectedNode={selectedNode}
        />
      </Canvas>
    </motion.div>
  )
}

export default NetworkGraph3D
